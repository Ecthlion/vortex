// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`WasmEncodingPlugin`] — a file-supplied array encoding backed by an embedded WebAssembly
//! kernel — and [`register_wasm_encodings`], which merges kernels into a session's encoding
//! registry.
//!
//! The plugin registers under the *encoding's own id* (e.g. `fastlanes.bitpacked`). Its
//! `deserialize` receives the node's real serialized parts and drives the kernel, returning the
//! decoded array — so wasm-backed encodings are decode-only and nothing wasm-specific survives
//! past deserialization.
//!
//! A kernel produces its output one of two ways. Value-producing encodings materialize bytes.
//! Re-arranging encodings (run-end, dict, ...) instead name a child and a gather, which the host
//! executes with `ArrayRef::take` — so the gathered child is never canonicalized, never copied
//! into the sandbox, and may have any dtype, including ones the kernel could not represent.
//!
//! Kernels never shadow native decoders: [`register_wasm_encodings`] skips any id already present
//! in the session registry.

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::session::ArraySession;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ChildMode;
use crate::KernelOutput;
use crate::WasmKernel;

/// An array encoding whose decoder is an embedded WebAssembly kernel.
pub struct WasmEncodingPlugin {
    id: ArrayId,
    kernel: Arc<WasmKernel>,
}

impl WasmEncodingPlugin {
    /// Create a plugin for `id` backed by an already-compiled kernel.
    pub fn new(id: impl Into<ArrayId>, kernel: Arc<WasmKernel>) -> Self {
        Self {
            id: id.into(),
            kernel,
        }
    }

    /// Compile `wasm_bytes` and create a plugin for `id`.
    pub fn try_new(id: impl Into<ArrayId>, wasm_bytes: impl AsRef<[u8]>) -> VortexResult<Self> {
        Ok(Self::new(id, Arc::new(WasmKernel::new(wasm_bytes)?)))
    }
}

impl ArrayPlugin for WasmEncodingPlugin {
    fn id(&self) -> ArrayId {
        self.id
    }

    fn is_supported_encoding(&self, _id: &ArrayId) -> bool {
        // Deserialization returns the *decoded* array, so the result legitimately carries a
        // canonical encoding id rather than this plugin's.
        true
    }

    fn serialize(
        &self,
        _array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Deserialization returns the decoded (canonical) array, so no array ever carries this
        // plugin's encoding id in memory; writing the encoding happens through the native VTable.
        vortex_bail!(
            "wasm-backed encoding {} is decode-only and cannot serialize",
            self.id
        )
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = session.create_execution_ctx();

        let buffers: Vec<ByteBuffer> = buffers
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<_>>()?;

        let mut decoder = self.kernel.decoder()?;

        // Ask the kernel which children the node has and how it intends to use each.
        let descriptors = decoder.children(dtype, len, children.len(), metadata)?;

        // Only `Values` children are decoded and copied into the sandbox. `Reference` children are
        // left alone here: they are resolved lazily below, in their own encoding, and only if the
        // kernel's result actually names one.
        let mut values_children = Vec::new();
        for (idx, d) in descriptors.iter().enumerate() {
            if d.mode == ChildMode::Values {
                values_children.push(
                    children
                        .get(idx, &d.dtype, d.len)?
                        .execute::<Canonical>(&mut ctx)?,
                );
            }
        }

        let output = decoder.decode(dtype, len, metadata, &buffers, &values_children, &mut ctx)?;

        let decoded = match output {
            KernelOutput::Materialized(array) => array,
            KernelOutput::Take {
                values_slot,
                indices,
            } => {
                let descriptor = descriptors.get(values_slot).ok_or_else(|| {
                    vortex_err!(
                        "wasm kernel for {} gathered child {values_slot}, but only {} were declared",
                        self.id,
                        descriptors.len()
                    )
                })?;
                vortex_ensure!(
                    descriptor.mode == ChildMode::Reference,
                    "wasm kernel for {} gathered child {values_slot}, which it declared as Values",
                    self.id
                );
                validate_indices(&indices, descriptor.len, len, &mut ctx)?;
                // Resolve the gathered child in its own encoding — never canonicalized, never
                // copied into the sandbox — and let Vortex gather it lazily.
                children
                    .get(values_slot, &descriptor.dtype, descriptor.len)?
                    .take(indices)?
            }
        };

        vortex_ensure!(
            decoded.len() == len,
            "wasm kernel for {} decoded {} rows, expected {len}",
            self.id,
            decoded.len()
        );
        vortex_ensure!(
            decoded.dtype() == dtype,
            "wasm kernel for {} decoded dtype {}, expected {dtype}",
            self.id,
            decoded.dtype()
        );
        Ok(decoded)
    }
}

/// Validate gather indices produced by an untrusted kernel.
///
/// Deliberately recomputes the maximum over the materialized index buffer rather than consulting
/// `Stat::Max`: statistics are themselves attacker-controlled file data, so they must never be
/// load-bearing for a safety property. This matters because `ArrayRef::take` builds a `DictArray`,
/// whose constructor checks only that the codes are integral — an out-of-range index would
/// otherwise surface later as a panic or garbage.
fn validate_indices(
    indices: &ArrayRef,
    values_len: usize,
    expected_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    vortex_ensure!(
        indices.len() == expected_len,
        "wasm kernel produced {} gather indices, expected {expected_len}",
        indices.len()
    );
    let DType::Primitive(ptype, _) = indices.dtype() else {
        vortex_bail!(
            "wasm gather indices must be primitive, got {}",
            indices.dtype()
        );
    };
    vortex_ensure!(
        ptype.is_unsigned_int(),
        "wasm gather indices must be an unsigned integer, got {ptype}"
    );
    vortex_ensure!(
        !indices.dtype().is_nullable(),
        "wasm gather indices must be non-nullable"
    );

    let primitive = indices.clone().execute::<Canonical>(ctx)?.into_primitive();
    let in_bounds = match_each_unsigned_integer_ptype!(primitive.ptype(), |P| {
        primitive
            .as_slice::<P>()
            .iter()
            .all(|&i| (i as u128) < values_len as u128)
    });
    vortex_ensure!(
        in_bounds,
        "wasm gather index out of bounds for a child of length {values_len}"
    );
    Ok(())
}

/// Merge embedded kernels into `session`'s array-encoding registry, returning the ids actually
/// registered.
///
/// A native encoding always supersedes a kernel: ids already present in the registry are skipped.
/// Kernels for genuinely unknown encodings are compiled and registered, so subsequent
/// deserialization of those encodings decodes through the sandboxed kernel.
pub fn register_wasm_encodings(
    session: &VortexSession,
    kernels: impl IntoIterator<Item = (String, ByteBuffer)>,
) -> VortexResult<Vec<String>> {
    let arrays = session.get::<ArraySession>();
    let mut registered = Vec::new();
    for (id, wasm_bytes) in kernels {
        let array_id = ArrayId::new(&id);
        if arrays.registry().find(&array_id).is_some() {
            // The reader has a native decoder for this encoding; it supersedes the kernel.
            continue;
        }
        arrays.register(WasmEncodingPlugin::try_new(
            array_id,
            wasm_bytes.as_slice(),
        )?);
        registered.push(id);
    }
    Ok(registered)
}
