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
//! A kernel does not return an array, it returns a *plan* — a small tree of operations over the
//! node's children that the host evaluates with its own lazy arrays.
//! Value-producing encodings end their plan in one materialized node. Re-arranging encodings
//! (run-end, dict, sparse, ...) name their children and say what to do with them, so those
//! children are never canonicalized, never copied into the sandbox, and may have any dtype —
//! including ones the kernel could not represent.
//!
//! Kernels never shadow native decoders: [`register_wasm_encodings`] skips any id already present
//! in the session registry.

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::VortexSessionExecute;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayChildren;
use vortex_array::session::ArraySession;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ChildMode;
use crate::WasmKernel;
use crate::plan::PlanContext;

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

        let (plan, guest_mem) =
            decoder.decode(dtype, len, metadata, &buffers, &values_children, &mut ctx)?;

        // Only the child slots the plan actually names get resolved, and each in its own encoding:
        // a `Reference` child a plan does not mention costs nothing at all.
        let id = self.id;
        let mut resolve = |slot: usize| -> VortexResult<ArrayRef> {
            let descriptor = descriptors.get(slot).ok_or_else(|| {
                vortex_err!(
                    "wasm kernel for {id} names child {slot}, but only {} were declared",
                    descriptors.len()
                )
            })?;
            vortex_ensure!(
                descriptor.mode == ChildMode::Reference,
                "wasm kernel for {id} names child {slot} in its plan, but declared it as Values"
            );
            children.get(slot, &descriptor.dtype, descriptor.len)
        };
        let decoded = plan.evaluate(
            &mut PlanContext {
                dtype,
                len,
                child: &mut resolve,
            },
            &mut ctx,
            &guest_mem,
        )?;

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
