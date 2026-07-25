// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoder kernels embedded in a Vortex file.
//!
//! A file may carry a portable decoder for an array encoding it uses, so a reader that was never
//! compiled against that encoding can still read the file. The kernels are ordinary segments
//! referenced from the postscript by encoding id, alongside the dtype, layout, statistics, and
//! footer segments.
//!
//! This crate deliberately knows nothing about how a kernel runs: it locates the bytes and hands
//! them to an [`EmbeddedKernelLoader`] installed on the session. `vortex-wasm` provides the
//! WebAssembly implementation. Two properties follow from that split:
//!
//! - A reader that has not installed a loader ignores embedded kernels entirely, and an encoding
//!   with no native decoder fails as an unknown encoding, exactly as it would today. Running
//!   file-supplied code is opt-in.
//! - A native decoder always supersedes a kernel. The reader never even fetches the bytes for an
//!   encoding it already knows how to decode.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

/// A decoder kernel embedded in a Vortex file, and the array encoding it decodes.
#[derive(Clone, Debug)]
pub struct EmbeddedKernel {
    id: String,
    abi_version: u32,
    module: ByteBuffer,
}

impl EmbeddedKernel {
    /// Create a kernel for the array encoding `id`.
    ///
    /// `abi_version` is the host/guest ABI the module was built against; a loader is expected to
    /// reject a version it does not implement.
    pub fn new(id: impl Into<String>, abi_version: u32, module: impl Into<ByteBuffer>) -> Self {
        Self {
            id: id.into(),
            abi_version,
            module: module.into(),
        }
    }

    /// The array encoding id this kernel decodes, e.g. `fastlanes.bitpacked`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The host/guest ABI version the kernel was built against.
    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }

    /// The kernel module's bytes.
    pub fn module(&self) -> &ByteBuffer {
        &self.module
    }
}

/// Turns kernels embedded in a file into array encodings a reader can use.
///
/// Implementations receive only the kernels for encodings the session cannot already decode, and
/// must return a session in which those encodings are registered. That session is scoped to the
/// one file being opened, so an implementation must not register into the session it was given —
/// see [`ArraySession::fork`](vortex_array::session::ArraySession::fork).
pub trait EmbeddedKernelLoader: Debug + Send + Sync + 'static {
    /// Return a session that can decode the encodings named by `kernels`.
    ///
    /// Returning an error fails the file open; a loader that wants to be permissive about kernels
    /// it cannot use should skip them and return a session instead.
    fn load(
        &self,
        session: &VortexSession,
        kernels: &[EmbeddedKernel],
    ) -> VortexResult<VortexSession>;
}

/// Session state controlling whether file-embedded decoder kernels are run.
///
/// Empty by default: embedded kernels are ignored unless a loader is installed.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use vortex_file::EmbeddedKernelSession;
/// # use vortex_session::SessionExt;
/// # fn install(session: &vortex_session::VortexSession, loader: Arc<dyn vortex_file::EmbeddedKernelLoader>) {
/// *session.get_mut::<EmbeddedKernelSession>() = EmbeddedKernelSession::new(loader);
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct EmbeddedKernelSession {
    loader: Option<Arc<dyn EmbeddedKernelLoader>>,
}

impl EmbeddedKernelSession {
    /// Install `loader` as the handler for kernels embedded in files opened with this session.
    pub fn new(loader: Arc<dyn EmbeddedKernelLoader>) -> Self {
        Self {
            loader: Some(loader),
        }
    }

    /// The installed loader, if any.
    pub fn loader(&self) -> Option<&Arc<dyn EmbeddedKernelLoader>> {
        self.loader.as_ref()
    }
}

impl SessionVar for EmbeddedKernelSession {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
