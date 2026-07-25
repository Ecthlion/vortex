// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::tests_outside_test_module)]

//! End-to-end tests of decoder kernels embedded in a **Vortex file**.
//!
//! `plugin_roundtrip.rs` covers the array level: serialized bytes in, decoded array out. These
//! tests cover the file level — the postscript's kernel segments, the fetch-only-what-you-need
//! rule, and registration at open — by writing real files and reading them back with a session
//! that has no native decoder for the encoding they use.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::session::ArraySession;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_fastlanes::BitPacked;
use vortex_file::EmbeddedKernel;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_wasm::abi::ABI_VERSION;
use vortex_wasm::embed_kernel;
use vortex_wasm::with_wasm_kernel_loader;

/// The `fastlanes.bitpacked` kernel (`encodings/fastlanes/wasm`).
const BITPACKED_KERNEL: &[u8] = include_bytes!("fixtures/bitpacked_kernel.wasm");
const BITPACKED_ID: &str = "fastlanes.bitpacked";

/// Small non-negative integers, which the compressor bit-packs.
static VALUES: LazyLock<Buffer<u32>> =
    LazyLock::new(|| Buffer::from_iter((0..4096u32).map(|i| i % 37)));

fn base_session() -> VortexSession {
    array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

/// A writer's session: has the native `fastlanes.bitpacked` encoding.
fn writer_session() -> VortexSession {
    let session = base_session();
    session.get::<ArraySession>().register(BitPacked);
    session
}

/// A reader's session: canonical encodings only, so `fastlanes.bitpacked` is unknown.
fn reader_session() -> VortexSession {
    base_session()
}

fn expected() -> ArrayRef {
    PrimitiveArray::new(VALUES.clone(), Validity::NonNullable).into_array()
}

/// Write a file of bit-packable integers, optionally embedding the bitpacked kernel.
async fn write_file(kernel: Option<EmbeddedKernel>) -> VortexResult<ByteBufferMut> {
    let mut options = writer_session().write_options();
    if let Some(kernel) = kernel {
        options = options.with_wasm_kernel(kernel);
    }
    let mut buf = ByteBufferMut::empty();
    // Two chunks so the file has a non-trivial layout over the encoded data.
    let chunked = ChunkedArray::from_iter([expected(), expected()]).into_array();
    options.write(&mut buf, chunked.to_array_stream()).await?;
    Ok(buf)
}

async fn scan_all(file: VortexFile) -> VortexResult<ArrayRef> {
    file.scan()?.into_array_stream()?.read_all().await
}

/// The premise of every test below: the writer really did use an encoding the reader lacks.
///
/// Encoding ids resolve lazily, so this surfaces during the scan rather than at open.
#[tokio::test]
async fn reader_without_the_encoding_cannot_read_the_file() -> VortexResult<()> {
    let buf = write_file(None).await?;

    let file = reader_session().open_options().open_buffer(buf.freeze())?;
    let error = scan_all(file)
        .await
        .expect_err("a reader without fastlanes.bitpacked should not be able to read the file");
    assert!(
        error.to_string().contains(BITPACKED_ID),
        "expected an unknown-encoding error naming {BITPACKED_ID}, got: {error}"
    );
    Ok(())
}

#[tokio::test]
async fn embedded_kernel_decodes_a_file_whose_encoding_the_reader_lacks() -> VortexResult<()> {
    let buf = write_file(Some(embed_kernel(BITPACKED_ID, BITPACKED_KERNEL)?)).await?;

    let session = with_wasm_kernel_loader(reader_session());
    let file = session.open_options().open_buffer(buf.freeze())?;
    let decoded = scan_all(file).await?;

    let expected = ChunkedArray::from_iter([expected(), expected()]).into_array();
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

/// Kernels are the reader's fallback, not a replacement: without a loader installed, a file's
/// kernels are ignored and the encoding stays unknown.
#[tokio::test]
async fn embedded_kernels_are_ignored_without_a_loader() -> VortexResult<()> {
    let buf = write_file(Some(embed_kernel(BITPACKED_ID, BITPACKED_KERNEL)?)).await?;

    let file = reader_session().open_options().open_buffer(buf.freeze())?;
    assert!(
        scan_all(file).await.is_err(),
        "embedded kernels must not run unless a loader is installed"
    );
    Ok(())
}

/// A reader that *has* the native encoding must keep using it, and must not be affected by
/// whatever the file happens to carry.
#[tokio::test]
async fn native_encoding_supersedes_an_embedded_kernel() -> VortexResult<()> {
    // A kernel registered under the bitpacked id, but which is not a bitpacked decoder at all.
    // If the reader ever ran it, the scan would produce garbage or fail.
    let wrong_kernel = EmbeddedKernel::new(
        BITPACKED_ID,
        ABI_VERSION,
        ByteBuffer::copy_from(include_bytes!("fixtures/runend_kernel.wasm")),
    );
    let buf = write_file(Some(wrong_kernel)).await?;

    let session = with_wasm_kernel_loader(writer_session());
    let file = session.open_options().open_buffer(buf.freeze())?;
    let decoded = scan_all(file).await?;

    let expected = ChunkedArray::from_iter([expected(), expected()]).into_array();
    let mut ctx = session.create_execution_ctx();
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

/// A file's kernels are scoped to that file: they must not leak into the session the reader
/// passed in, or a second file using the same encoding id would silently be decoded by the first
/// file's code.
#[tokio::test]
async fn kernels_do_not_leak_into_the_callers_session() -> VortexResult<()> {
    let buf = write_file(Some(embed_kernel(BITPACKED_ID, BITPACKED_KERNEL)?)).await?;

    let session = with_wasm_kernel_loader(reader_session());
    let file = session.open_options().open_buffer(buf.freeze())?;
    scan_all(file).await?;

    assert!(
        session
            .get::<ArraySession>()
            .registry()
            .find(&ArrayId::new(BITPACKED_ID))
            .is_none(),
        "opening a file must not register its kernels on the caller's session"
    );
    Ok(())
}

/// The postscript records each kernel's ABI version so a reader can reject one it cannot drive
/// before compiling or running any of it.
#[tokio::test]
async fn mismatched_abi_version_is_rejected() -> VortexResult<()> {
    let stale = EmbeddedKernel::new(
        BITPACKED_ID,
        ABI_VERSION + 1,
        ByteBuffer::copy_from(BITPACKED_KERNEL),
    );
    let buf = write_file(Some(stale)).await?;

    let session = with_wasm_kernel_loader(reader_session());
    let error = session
        .open_options()
        .open_buffer(buf.freeze())
        .err()
        .expect("a kernel targeting a different ABI version must be rejected");
    assert!(
        error.to_string().contains("ABI version"),
        "expected an ABI version error, got: {error}"
    );
    Ok(())
}

/// A reader that already has the encoding never even fetches the kernel bytes.
///
/// Differential: the same file read by a session that *needs* the kernel does pull it in, so the
/// byte counts below distinguish the two paths rather than merely being small.
#[tokio::test]
async fn a_native_reader_does_not_read_the_kernel_segment() -> VortexResult<()> {
    /// Padding to make a fetch of the kernel unmistakable next to the ~64 KiB tail read.
    const PADDING: usize = 1 << 20;

    let buf = write_file(Some(EmbeddedKernel::new(
        BITPACKED_ID,
        ABI_VERSION,
        ByteBuffer::copy_from(pad_module(BITPACKED_KERNEL, PADDING)),
    )))
    .await?
    .freeze();

    async fn bytes_read_by(session: VortexSession, buf: ByteBuffer) -> VortexResult<u64> {
        let reads = Arc::new(AtomicU64::new(0));
        let source = CountingReadAt {
            buffer: buf,
            bytes_read: Arc::clone(&reads),
        };
        let file = session.open_options().open(Arc::new(source)).await?;
        scan_all(file).await?;
        Ok(reads.load(Ordering::Relaxed))
    }

    let native = bytes_read_by(with_wasm_kernel_loader(writer_session()), buf.clone()).await?;
    let via_kernel = bytes_read_by(with_wasm_kernel_loader(reader_session()), buf).await?;

    assert!(
        native < PADDING as u64,
        "a reader with the native encoding read {native} bytes, which means it fetched the kernel"
    );
    assert!(
        via_kernel > PADDING as u64,
        "a reader relying on the kernel read only {via_kernel} bytes, so this test proves nothing"
    );
    Ok(())
}

/// Grow a wasm module by `padding` bytes while keeping it valid.
///
/// Appending raw bytes would not do: wasmtime rejects trailing garbage. A custom section is the
/// module format's own extension point, and is ignored by everything that does not know its name.
fn pad_module(module: &[u8], padding: usize) -> Vec<u8> {
    fn leb128(len: usize, out: &mut Vec<u8>) {
        let mut value = len;
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                return;
            }
        }
    }

    const NAME: &[u8] = b"vortex.test.padding";
    let mut body = Vec::with_capacity(padding + NAME.len() + 8);
    leb128(NAME.len(), &mut body);
    body.extend_from_slice(NAME);
    body.resize(body.len() + padding, 0);

    let mut out = module.to_vec();
    out.push(0); // Custom section id.
    leb128(body.len(), &mut out);
    out.extend_from_slice(&body);
    out
}

/// A [`VortexReadAt`] over an in-memory buffer that counts the bytes it is asked for.
struct CountingReadAt {
    buffer: ByteBuffer,
    bytes_read: Arc<AtomicU64>,
}

impl VortexReadAt for CountingReadAt {
    fn concurrency(&self) -> usize {
        1
    }

    fn size(&self) -> futures::future::BoxFuture<'static, VortexResult<u64>> {
        let size = self.buffer.len() as u64;
        Box::pin(async move { Ok(size) })
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> futures::future::BoxFuture<'static, VortexResult<BufferHandle>> {
        self.bytes_read.fetch_add(length as u64, Ordering::Relaxed);
        let buffer = self.buffer.clone();
        Box::pin(async move {
            let start = usize::try_from(offset)?;
            let end = start + length;
            if end > buffer.len() {
                vortex_bail!(
                    "read {start}..{end} out of bounds for {} bytes",
                    buffer.len()
                );
            }
            Ok(BufferHandle::new_host(
                buffer.slice_unaligned(start..end).aligned(alignment),
            ))
        })
    }
}
