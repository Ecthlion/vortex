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
//! A kernel always materializes its output: it returns a canonical array of the node's own dtype,
//! whatever that dtype is. Children the kernel declares are decoded by the host (natively, or
//! recursively through another kernel), canonicalized, and copied into the sandbox, so a kernel
//! that re-arranges a child (run-end, dict, sparse, ...) gathers it in-sandbox with a generic
//! `take` that works for any dtype.
//!
//! Kernels never shadow native decoders: [`register_wasm_encodings`] skips any id already present
//! in the session registry.

use std::sync::Arc;

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::VortexSessionExecute;
use vortex_array::session::ArraySession;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

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
    ) -> VortexResult<Option<ArraySerialization>> {
        // Deserialization returns the decoded (canonical) array, so no array ever carries this
        // plugin's encoding id in memory; writing the encoding happens through the native VTable.
        vortex_bail!(
            "wasm-backed encoding {} is decode-only and cannot serialize",
            self.id
        )
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        // The plugin is registered under exactly one wire id, so `serialized_id` is always ours.
        let ArrayDeserialization {
            serialized_id: _,
            dtype,
            len,
            metadata,
            buffers,
            children,
        } = parts;
        let mut ctx = session.create_execution_ctx();

        let buffers: Vec<ByteBuffer> = buffers
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<_>>()?;

        let mut decoder = self.kernel.decoder()?;

        // Ask the kernel for the dtype and length of every serialized child, then decode each one
        // in its own encoding. The kernel receives them canonicalized.
        let descriptors = decoder.children(dtype, len, children.len(), metadata, session)?;
        let decoded_children = descriptors
            .iter()
            .enumerate()
            .map(|(idx, d)| children.get(idx, &d.dtype, d.len))
            .collect::<VortexResult<Vec<ArrayRef>>>()?;

        let decoded =
            decoder.decode(dtype, len, metadata, &buffers, &decoded_children, &mut ctx)?;

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
        let array_id = ArrayId::from(id.as_str());
        if arrays.registry().contains_key(&array_id) {
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
