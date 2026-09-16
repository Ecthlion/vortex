// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use futures::poll;
use parking_lot::Mutex;
use rstest::rstest;
use tokio::sync::Semaphore;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::oneshot;
use tokio::time::timeout;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;

use super::DEFAULT_FILE_CONCURRENCY;
use super::FILE_READ_CHUNK_BYTES;
use super::FileReadBackend;
use super::PooledFileReadAt;
use super::PooledFileReadAtOptions;
use super::PooledHostRead;
use crate::CudaSessionExt;
use crate::pinned::PinnedByteBufferPool;

const WAIT: Duration = Duration::from_secs(30);
const FILE_OFFSET: u64 = (1 << 32) + 37;

#[derive(Default)]
struct FakeFileReadBackend {
    padded: bool,
    eof: Option<u64>,
    requests: Mutex<Vec<(u64, usize)>>,
    blocked: Option<UnboundedSender<BlockedRead>>,
}

struct BlockedRead {
    offset: u64,
    release: mpsc::Sender<()>,
    finished: oneshot::Receiver<()>,
}

impl BlockedRead {
    async fn finish(self) -> VortexResult<()> {
        self.release
            .send(())
            .map_err(|error| vortex_err!("failed to release fake read: {error}"))?;
        timeout(WAIT, self.finished)
            .await
            .map_err(|error| vortex_err!("fake read did not finish: {error}"))?
            .map_err(|error| vortex_err!("fake read dropped completion: {error}"))?;
        Ok(())
    }
}

impl FileReadBackend for FakeFileReadBackend {
    fn size(&self) -> VortexResult<u64> {
        Ok(self.eof.unwrap_or(u64::MAX))
    }

    fn read(
        &self,
        pool: &Arc<PinnedByteBufferPool>,
        offset: u64,
        length: usize,
    ) -> VortexResult<PooledHostRead> {
        self.requests.lock().push((offset, length));
        let end = offset
            .checked_add(u64::try_from(length)?)
            .ok_or_else(|| vortex_err!("overflow reached fake backend"))?;
        let prefix = if self.padded {
            usize::try_from(offset % 4096)?
        } else {
            0
        };
        let source_len = if self.padded {
            (prefix + length).next_multiple_of(4096)
        } else {
            length
        };
        let mut buffer = pool.get(source_len)?;
        buffer.as_mut_slice().fill(0xFF);
        for (index, byte) in buffer.as_mut_slice()[prefix..prefix + length]
            .iter_mut()
            .enumerate()
        {
            *byte = file_byte(offset + index as u64);
        }

        // Block only after acquiring the buffer, so cancellation also exercises its ownership.
        let finished = if let Some(blocked) = &self.blocked {
            let (release, wait) = mpsc::channel();
            let (finished, completion) = oneshot::channel();
            blocked
                .send(BlockedRead {
                    offset,
                    release,
                    finished: completion,
                })
                .map_err(|_| vortex_err!("fake read controller dropped"))?;
            wait.recv_timeout(WAIT)
                .map_err(|error| vortex_err!("fake read was not released: {error}"))?;
            Some(finished)
        } else {
            None
        };

        let result = if self.eof.is_some_and(|eof| end > eof) {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("fake file short read at {offset}: requested {length} bytes"),
            )
            .into())
        } else {
            Ok(PooledHostRead {
                buffer,
                requested_range: prefix..prefix + length,
            })
        };
        if let Some(finished) = finished {
            let _ = finished.send(());
        }
        result
    }
}

fn file_byte(offset: u64) -> u8 {
    // Unlike a repeating u8 counter, this distinguishes adjacent 4 MiB chunks.
    (offset % 251) as u8
}

fn reader(
    backend: Arc<FakeFileReadBackend>,
) -> VortexResult<(CurrentThreadRuntime, PooledFileReadAt)> {
    let runtime = CurrentThreadRuntime::new();
    let session = crate::cuda_session().with_handle(runtime.handle());
    let cuda = session.cuda_session();
    let reader = PooledFileReadAt {
        uri: Arc::from("fake-file"),
        backend,
        handle: session.handle(),
        pool: Arc::clone(cuda.pinned_buffer_pool()),
        stream: cuda.stream()?,
        read_slots: Arc::new(Semaphore::new(DEFAULT_FILE_CONCURRENCY)),
    };
    Ok((runtime, reader))
}

async fn next_read(blocked: &mut UnboundedReceiver<BlockedRead>) -> VortexResult<BlockedRead> {
    timeout(WAIT, blocked.recv())
        .await
        .map_err(|error| vortex_err!("fake read did not start: {error}"))?
        .ok_or_else(|| vortex_err!("fake read backend dropped"))
}

async fn assert_bytes(buffer: BufferHandle, offset: u64, length: usize) -> VortexResult<()> {
    assert!(buffer.is_on_device());
    assert_eq!(buffer.len(), length);
    let host = buffer.try_to_host()?.await?;
    let expected: Vec<_> = (0..length)
        .map(|index| file_byte(offset + index as u64))
        .collect();
    assert_eq!(host.as_ref(), expected.as_slice());
    Ok(())
}

#[test]
fn pooled_file_read_options() {
    assert!(!PooledFileReadAtOptions::default().direct_io);
    #[cfg(target_os = "linux")]
    assert!(
        PooledFileReadAtOptions::default()
            .with_direct_io()
            .direct_io
    );
}

#[rstest]
#[case::single_buffered(FILE_READ_CHUNK_BYTES - 1, false)]
#[case::single_padded(FILE_READ_CHUNK_BYTES, true)]
#[case::exact_multichunk(2 * FILE_READ_CHUNK_BYTES, false)]
#[case::multichunk_tail(2 * FILE_READ_CHUNK_BYTES + 19, true)]
#[crate::test]
async fn chunks_preserve_bytes_and_offsets_in_reverse_completion_order(
    #[case] length: usize,
    #[case] padded: bool,
) -> VortexResult<()> {
    let (send, mut blocked) = unbounded_channel();
    let backend = Arc::new(FakeFileReadBackend {
        padded,
        blocked: Some(send),
        ..Default::default()
    });
    let (_runtime, reader) = reader(Arc::clone(&backend))?;
    let mut read = reader.read_at(FILE_OFFSET, length, Alignment::of::<u8>());
    assert!(poll!(&mut read).is_pending());

    let mut chunks = Vec::new();
    let mut expected = Vec::new();
    for start in (0..length).step_by(FILE_READ_CHUNK_BYTES) {
        chunks.push(next_read(&mut blocked).await?);
        expected.push((
            FILE_OFFSET + start as u64,
            (length - start).min(FILE_READ_CHUNK_BYTES),
        ));
    }
    let mut requests = backend.requests.lock().clone();
    requests.sort_unstable();
    assert_eq!(requests, expected);
    chunks.sort_unstable_by_key(|chunk| chunk.offset);
    for chunk in chunks.into_iter().rev() {
        chunk.finish().await?;
    }

    let buffer = timeout(WAIT, read)
        .await
        .map_err(|error| vortex_err!("chunked read did not finish: {error}"))??;
    assert_bytes(buffer, FILE_OFFSET, length).await
}

#[crate::test]
async fn short_read_propagates_without_waiting_for_an_earlier_chunk() -> VortexResult<()> {
    let (send, mut blocked) = unbounded_channel();
    let tail_offset = FILE_OFFSET + FILE_READ_CHUNK_BYTES as u64;
    let backend = Arc::new(FakeFileReadBackend {
        eof: Some(tail_offset + 16),
        blocked: Some(send),
        ..Default::default()
    });
    let (_runtime, mut reader) = reader(backend)?;
    reader.read_slots = Arc::new(Semaphore::new(2));
    let mut read = reader.read_at(
        FILE_OFFSET,
        FILE_READ_CHUNK_BYTES + 17,
        Alignment::of::<u8>(),
    );
    assert!(poll!(&mut read).is_pending());
    let first = next_read(&mut blocked).await?;
    let second = next_read(&mut blocked).await?;
    let (head, tail) = if first.offset == FILE_OFFSET {
        (first, second)
    } else {
        (second, first)
    };
    assert_eq!(tail.offset, tail_offset);
    tail.finish().await?;

    let result = timeout(WAIT, read)
        .await
        .map_err(|error| vortex_err!("short read waited for the blocked head: {error}"))?;
    let Err(error) = result else {
        vortex_bail!("a short chunk must fail the whole read");
    };
    assert!(error.to_string().contains("fake file short read"));
    assert!(error.to_string().contains(&tail_offset.to_string()));
    assert_eq!(reader.read_slots.available_permits(), 1);

    head.finish().await?;
    let permits = timeout(WAIT, reader.read_slots.acquire_many(2))
        .await
        .map_err(|error| vortex_err!("short read leaked a read slot: {error}"))?
        .map_err(|error| vortex_err!("read slots closed: {error}"))?;
    drop(permits);
    Ok(())
}

#[rstest]
#[case::single_chunk(16)]
#[case::multichunk(FILE_READ_CHUNK_BYTES + 1)]
#[case::extreme_length(usize::MAX)]
#[crate::test]
async fn overflowing_range_fails_before_reading(#[case] length: usize) -> VortexResult<()> {
    let backend = Arc::new(FakeFileReadBackend::default());
    let (_runtime, reader) = reader(Arc::clone(&backend))?;
    let allocations = reader.pool.stats().allocs;
    reader.read_slots.close();
    let Err(error) = reader
        .read_at(u64::MAX - 7, length, Alignment::of::<u8>())
        .await
    else {
        vortex_bail!("overflow must be rejected before acquiring a read slot");
    };
    assert!(error.to_string().contains("file read range overflow"));
    assert!(backend.requests.lock().is_empty());
    assert_eq!(reader.pool.stats().allocs, allocations);
    Ok(())
}

#[crate::test]
async fn cancelled_reads_keep_slots_and_buffers_until_blocking_io_finishes() -> VortexResult<()> {
    let (send, mut blocked) = unbounded_channel();
    let backend = Arc::new(FakeFileReadBackend {
        blocked: Some(send),
        ..Default::default()
    });
    let (_runtime, mut reader) = reader(Arc::clone(&backend))?;
    // Exercise the shared bound without allocating DEFAULT_FILE_CONCURRENCY full chunks.
    reader.read_slots = Arc::new(Semaphore::new(2));
    let mut cancelled = reader.read_at(
        FILE_OFFSET,
        2 * FILE_READ_CHUNK_BYTES + 11,
        Alignment::of::<u8>(),
    );
    assert!(poll!(&mut cancelled).is_pending());
    let first = next_read(&mut blocked).await?;
    let second = next_read(&mut blocked).await?;
    assert_eq!(reader.read_slots.available_permits(), 0);

    let replacement_offset = FILE_OFFSET + 9 * FILE_READ_CHUNK_BYTES as u64;
    let mut replacement = reader
        .clone()
        .read_at(replacement_offset, 31, Alignment::of::<u8>());
    assert!(poll!(&mut replacement).is_pending());
    drop(cancelled);
    assert_eq!(reader.read_slots.available_permits(), 0);
    assert_eq!(reader.pool.stats().puts, 0);
    assert!(poll!(&mut replacement).is_pending());
    assert_eq!(backend.requests.lock().len(), 2);

    first.finish().await?;
    let resumed = tokio::select! {
        resumed = next_read(&mut blocked) => resumed?,
        result = &mut replacement => {
            result?;
            vortex_bail!("replacement completed before its backend was released");
        }
    };
    assert_eq!(resumed.offset, replacement_offset);
    assert_eq!(reader.read_slots.available_permits(), 0);
    second.finish().await?;
    resumed.finish().await?;
    let buffer = timeout(WAIT, replacement)
        .await
        .map_err(|error| vortex_err!("replacement read did not finish: {error}"))??;
    assert_bytes(buffer, replacement_offset, 31).await?;

    let permits = timeout(WAIT, reader.read_slots.acquire_many(2))
        .await
        .map_err(|error| vortex_err!("cancellation leaked a read slot: {error}"))?
        .map_err(|error| vortex_err!("read slots closed: {error}"))?;
    drop(permits);
    // The cancelled request's third chunk must never reach the backend.
    assert_eq!(backend.requests.lock().len(), 3);
    Ok(())
}
