// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Wiring kernels embedded in a Vortex file into the reader that opens it.
//!
//! `vortex-file` locates the kernel segments and decides which are needed; this module compiles
//! them and registers them as array encodings. The split keeps `wasmtime` out of the file crate,
//! and makes running file-supplied code opt-in: a reader that never installs
//! [`WasmKernelLoader`] ignores embedded kernels entirely.

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::session::ArraySession;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::EmbeddedKernel;
use vortex_file::EmbeddedKernelLoader;
use vortex_file::EmbeddedKernelSession;
use vortex_session::VortexSession;

use crate::WasmEncodingPlugin;
use crate::WasmKernel;
use crate::abi::ABI_VERSION;

/// Runs WebAssembly decoder kernels embedded in Vortex files.
///
/// Install it on a session with [`with_wasm_kernel_loader`], after which opening a file that
/// embeds kernels for encodings this reader lacks will decode those encodings through the
/// sandboxed kernels.
#[derive(Debug, Default, Clone, Copy)]
pub struct WasmKernelLoader;

impl EmbeddedKernelLoader for WasmKernelLoader {
    fn load(
        &self,
        session: &VortexSession,
        kernels: &[EmbeddedKernel],
    ) -> VortexResult<VortexSession> {
        // The file's encodings must not leak into the caller's session: another file may use the
        // same encoding id with a different kernel, and a kernel is untrusted file data. Forking
        // scopes the registrations to this file.
        let arrays = session.get::<ArraySession>().fork();

        for kernel in kernels {
            // `vortex-file` only passes us kernels for encodings the reader cannot already decode,
            // so an unusable one means this file is unreadable — say so here, where the reason is
            // known, rather than as an "unknown encoding" from deep inside a scan.
            if kernel.abi_version() != ABI_VERSION {
                vortex_bail!(
                    "embedded kernel for {} targets wasm ABI version {}, but this reader implements {ABI_VERSION}",
                    kernel.id(),
                    kernel.abi_version()
                );
            }
            arrays.register(WasmEncodingPlugin::try_new(
                ArrayId::new(kernel.id()),
                kernel.module().as_slice(),
            )?);
        }

        let mut builder = session.to_builder();
        *builder.get_mut::<ArraySession>() = arrays;
        Ok(builder.build())
    }
}

/// Install [`WasmKernelLoader`] on `session`, so files opened with it may supply their own
/// decoders.
pub fn with_wasm_kernel_loader(session: VortexSession) -> VortexSession {
    let mut builder = session.to_builder();
    *builder.get_mut::<EmbeddedKernelSession>() =
        EmbeddedKernelSession::new(Arc::new(WasmKernelLoader));
    builder.build()
}

/// Prepare a `.wasm` module for embedding in a Vortex file as the decoder for encoding `id`.
///
/// Compiles the module so that a kernel this host could not run is rejected at write time rather
/// than by whoever reads the file, and records the ABI version it declares.
pub fn embed_kernel(
    id: impl Into<String>,
    module: impl AsRef<[u8]>,
) -> VortexResult<EmbeddedKernel> {
    let compiled = WasmKernel::new(module.as_ref())?;
    Ok(EmbeddedKernel::new(
        id,
        compiled.abi_version(),
        vortex_buffer::ByteBuffer::copy_from(module.as_ref()),
    ))
}
