// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(unix, windows))]
use std::fs::File;
use std::io;
use std::sync::Arc;
#[cfg(any(unix, windows))]
use std::sync::atomic::AtomicU64;
#[cfg(any(unix, windows))]
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
#[cfg(any(unix, windows))]
use object_store::ObjectMeta;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
#[cfg(any(unix, windows))]
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
#[cfg(any(unix, windows))]
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// An object store backed I/O source.
pub struct ObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: HostAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

/// An object-store reader that promotes an exactly identified file payload into a persistent
/// positional reader.
#[cfg(any(unix, windows))]
pub struct PromotingObjectStoreReadAt {
    inner: ObjectStoreReadAt,
    promotion: Arc<FilePayloadPromotion>,
}

#[cfg(any(unix, windows))]
enum FilePayloadPromotionState {
    Eligible,
    File(Arc<File>),
    NonFile,
}

#[cfg(any(unix, windows))]
struct FilePayloadPromotion {
    expected_meta: ObjectMeta,
    state: Mutex<FilePayloadPromotionState>,
    diagnostics: bool,
    get_opts_calls: AtomicU64,
    file_payloads: AtomicU64,
    cache_hits: AtomicU64,
    installs: AtomicU64,
    install_races: AtomicU64,
    identity_mismatches: AtomicU64,
    blocking_read_submissions: AtomicU64,
    non_file_payloads: AtomicU64,
}

#[cfg(any(unix, windows))]
impl FilePayloadPromotion {
    fn new(expected_meta: ObjectMeta, diagnostics: bool) -> Self {
        Self {
            expected_meta,
            state: Mutex::new(FilePayloadPromotionState::Eligible),
            diagnostics,
            get_opts_calls: AtomicU64::new(0),
            file_payloads: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            installs: AtomicU64::new(0),
            install_races: AtomicU64::new(0),
            identity_mismatches: AtomicU64::new(0),
            blocking_read_submissions: AtomicU64::new(0),
            non_file_payloads: AtomicU64::new(0),
        }
    }

    fn cached_file(&self) -> Option<Arc<File>> {
        let file = match &*self.state.lock() {
            FilePayloadPromotionState::File(file) => Some(Arc::clone(file)),
            FilePayloadPromotionState::Eligible | FilePayloadPromotionState::NonFile => None,
        };
        if file.is_some() {
            self.record(&self.cache_hits, "cache_hit");
        }
        file
    }

    fn observe_file(&self, metadata: &ObjectMeta, file: File) -> VortexResult<Arc<File>> {
        self.record(&self.file_payloads, "file_payload");
        let candidate = Arc::new(file);
        let mut state = self.state.lock();
        match &*state {
            FilePayloadPromotionState::Eligible => {
                if metadata != &self.expected_meta {
                    self.record(&self.identity_mismatches, "identity_mismatch");
                    vortex_bail!(
                        "object identity changed while opening {} for a persistent positional reader",
                        self.expected_meta.location
                    );
                }
                *state = FilePayloadPromotionState::File(Arc::clone(&candidate));
                self.record(&self.installs, "install");
                Ok(candidate)
            }
            FilePayloadPromotionState::File(file) => {
                // Another exact-identity response already installed the authoritative descriptor.
                // Prefer it even if this concurrently opened response observed a replacement.
                self.record(&self.install_races, "install_race");
                Ok(Arc::clone(file))
            }
            FilePayloadPromotionState::NonFile => {
                if metadata != &self.expected_meta {
                    self.record(&self.identity_mismatches, "identity_mismatch");
                    vortex_bail!(
                        "object identity changed while opening {} for a persistent positional reader",
                        self.expected_meta.location
                    );
                }
                Ok(candidate)
            }
        }
    }

    fn observe_non_file(&self) {
        self.record(&self.non_file_payloads, "non_file_payload");
        let mut state = self.state.lock();
        if matches!(*state, FilePayloadPromotionState::Eligible) {
            *state = FilePayloadPromotionState::NonFile;
        }
    }

    fn record_identity_mismatch(&self) {
        self.record(&self.identity_mismatches, "identity_mismatch");
    }

    fn record(&self, counter: &AtomicU64, event: &'static str) {
        if self.diagnostics {
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::trace!(
                target: "vortex_io::file_payload_promotion",
                event,
                path = %self.expected_meta.location,
                "persistent file-payload promotion event"
            );
        }
    }
}

#[cfg(any(unix, windows))]
impl Drop for FilePayloadPromotion {
    fn drop(&mut self) {
        if !self.diagnostics {
            return;
        }
        tracing::debug!(
            target: "vortex_io::file_payload_promotion",
            path = %self.expected_meta.location,
            get_opts_calls = self.get_opts_calls.load(Ordering::Relaxed),
            file_payloads = self.file_payloads.load(Ordering::Relaxed),
            cache_hits = self.cache_hits.load(Ordering::Relaxed),
            installs = self.installs.load(Ordering::Relaxed),
            install_races = self.install_races.load(Ordering::Relaxed),
            identity_mismatches = self.identity_mismatches.load(Ordering::Relaxed),
            blocking_read_submissions = self.blocking_read_submissions.load(Ordering::Relaxed),
            non_file_payloads = self.non_file_payloads.load(Ordering::Relaxed),
            "persistent file-payload promotion counters"
        );
    }
}

impl ObjectStoreReadAt {
    /// Create a new object store source.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self::new_with_allocator(store, path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Create a new object store source with a custom writable buffer allocator.
    pub fn new_with_allocator(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
        }
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }

    /// Reuse an exactly identified file payload as a persistent positional reader.
    #[cfg(any(unix, windows))]
    pub fn into_file_payload_promoting(
        self,
        expected_meta: ObjectMeta,
        diagnostics: bool,
    ) -> VortexResult<PromotingObjectStoreReadAt> {
        if self.path != expected_meta.location {
            vortex_bail!(
                "persistent file-payload promotion path {} does not match expected object {}",
                self.path,
                expected_meta.location
            );
        }
        Ok(PromotingObjectStoreReadAt {
            inner: self,
            promotion: Arc::new(FilePayloadPromotion::new(expected_meta, diagnostics)),
        })
    }
}

async fn read_object_store_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    allocator: HostAllocatorRef,
    request: ReadAtRequest,
) -> VortexResult<BufferHandle> {
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    let range = offset..(offset + length as u64);
    let mut buffer = allocator.allocate(length, alignment)?;

    let response = store
        .get_opts(
            &path,
            GetOptions {
                range: Some(GetRange::Bounded(range.clone())),
                ..Default::default()
            },
        )
        .await?;

    let buffer = match response.payload {
        #[cfg(not(target_arch = "wasm32"))]
        GetResultPayload::File(file, _) => io_handle
            .spawn_blocking(move || {
                read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                Ok::<_, io::Error>(buffer)
            })
            .await
            .map_err(io::Error::other)?,
        #[cfg(target_arch = "wasm32")]
        GetResultPayload::File(..) => {
            unreachable!("File payload not supported on wasm32")
        }
        GetResultPayload::Stream(mut byte_stream) => {
            let mut written = 0usize;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                let end = written + bytes.len();
                vortex_ensure!(
                    end <= length,
                    "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                    end,
                    length,
                    range
                );
                buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                written = end;
            }

            vortex_ensure!(
                written == length,
                "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                written,
                length,
                range
            );

            buffer
        }
    };

    Ok(BufferHandle::new_host(buffer.freeze()))
}

#[cfg(any(unix, windows))]
async fn read_promoting_object_store_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    allocator: HostAllocatorRef,
    request: ReadAtRequest,
    file_payload_promotion: Arc<FilePayloadPromotion>,
) -> VortexResult<BufferHandle> {
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    let end = offset
        .checked_add(u64::try_from(length)?)
        .ok_or_else(|| vortex_err!("positional read range overflow"))?;
    let range = offset..end;

    if let Some(file) = file_payload_promotion.cached_file() {
        file_payload_promotion.record(
            &file_payload_promotion.blocking_read_submissions,
            "blocking_read_submission",
        );
        let mut buffer = allocator.allocate(length, alignment)?;
        let buffer = io_handle
            .spawn_blocking(move || {
                read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                Ok::<_, io::Error>(buffer)
            })
            .await
            .map_err(io::Error::other)?;
        return Ok(BufferHandle::new_host(buffer.freeze()));
    }

    let mut buffer = allocator.allocate(length, alignment)?;

    file_payload_promotion.record(&file_payload_promotion.get_opts_calls, "get_opts");

    let mut get_options = GetOptions {
        range: Some(GetRange::Bounded(range.clone())),
        ..Default::default()
    };
    get_options.if_match = file_payload_promotion.expected_meta.e_tag.clone();
    get_options.version = file_payload_promotion.expected_meta.version.clone();
    let response = store.get_opts(&path, get_options).await?;

    if response.range != range {
        file_payload_promotion.record_identity_mismatch();
        vortex_bail!(
            "object store returned range {:?} for requested persistent file range {:?}",
            response.range,
            range
        );
    }
    let response_meta = response.meta.clone();
    let buffer = match response.payload {
        GetResultPayload::File(file, _) => {
            let file = file_payload_promotion.observe_file(&response_meta, file)?;
            file_payload_promotion.record(
                &file_payload_promotion.blocking_read_submissions,
                "blocking_read_submission",
            );
            io_handle
                .spawn_blocking(move || {
                    read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                    Ok::<_, io::Error>(buffer)
                })
                .await
                .map_err(io::Error::other)?
        }
        GetResultPayload::Stream(mut byte_stream) => {
            if response_meta != file_payload_promotion.expected_meta {
                file_payload_promotion.record_identity_mismatch();
                vortex_bail!(
                    "object identity changed while opening {} for a persistent positional reader",
                    file_payload_promotion.expected_meta.location
                );
            }
            let mut written = 0usize;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes?;
                let end = written + bytes.len();
                vortex_ensure!(
                    end <= length,
                    "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                    end,
                    length,
                    range
                );
                buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                written = end;
            }

            vortex_ensure!(
                written == length,
                "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                written,
                length,
                range
            );

            file_payload_promotion.observe_non_file();
            buffer
        }
    };

    Ok(BufferHandle::new_host(buffer.freeze()))
}

impl VortexReadAt for ObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        let io_handle = handle.clone();
        handle
            .spawn_io(read_object_store_range(
                store,
                path,
                io_handle,
                allocator,
                ReadAtRequest::new(offset, length, alignment),
            ))
            .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        let concurrency = self.concurrency.max(1);
        let (mut send, recv) = mpsc::channel(concurrency);
        let io_handle = handle.clone();

        // A single runtime task drives all GETs, avoiding one spawn per range. Do not use
        // ObjectStore::get_ranges here: it returns one Vec after every range completes, whereas
        // VortexReadAt::read_ranges must expose each result as soon as it is ready.
        let task = handle.spawn_io(async move {
            let reads = requests.iter().copied().map(|request| {
                let store = Arc::clone(&store);
                let path = path.clone();
                let io_handle = io_handle.clone();
                let allocator = Arc::clone(&allocator);
                async move {
                    let result =
                        read_object_store_range(store, path, io_handle, allocator, request).await;
                    (request, result)
                }
            });

            let mut reads = stream::iter(reads).buffer_unordered(concurrency);
            while let Some(result) = reads.next().await {
                if send.send(result).await.is_err() {
                    break;
                }
            }
        });

        async_stream::stream! {
            let mut recv = recv;
            while let Some(result) = recv.next().await {
                yield result;
            }
            task.await;
        }
        .boxed()
    }
}

#[cfg(any(unix, windows))]
impl VortexReadAt for PromotingObjectStoreReadAt {
    fn diagnostic_instance_id(&self) -> Option<u64> {
        self.inner.diagnostic_instance_id()
    }

    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.promotion.expected_meta.size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.inner.store);
        let path = self.inner.path.clone();
        let handle = self.inner.handle.clone();
        let allocator = Arc::clone(&self.inner.allocator);
        let promotion = Arc::clone(&self.promotion);
        let io_handle = handle.clone();
        handle
            .spawn_io(read_promoting_object_store_range(
                store,
                path,
                io_handle,
                allocator,
                ReadAtRequest::new(offset, length, alignment),
                promotion,
            ))
            .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let store = Arc::clone(&self.inner.store);
        let path = self.inner.path.clone();
        let handle = self.inner.handle.clone();
        let allocator = Arc::clone(&self.inner.allocator);
        let promotion = Arc::clone(&self.promotion);
        let concurrency = self.inner.concurrency.max(1);
        let (mut send, recv) = mpsc::channel(concurrency);
        let io_handle = handle.clone();

        let task = handle.spawn_io(async move {
            let reads = requests.iter().copied().map(|request| {
                let store = Arc::clone(&store);
                let path = path.clone();
                let io_handle = io_handle.clone();
                let allocator = Arc::clone(&allocator);
                let promotion = Arc::clone(&promotion);
                async move {
                    let result = read_promoting_object_store_range(
                        store, path, io_handle, allocator, request, promotion,
                    )
                    .await;
                    (request, result)
                }
            });

            let mut reads = stream::iter(reads).buffer_unordered(concurrency);
            while let Some(result) = reads.next().await {
                if send.send(result).await.is_err() {
                    break;
                }
            }
        });

        async_stream::stream! {
            let mut recv = recv;
            while let Some(result) = recv.next().await {
                yield result;
            }
            task.await;
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {

    use std::collections::VecDeque;
    use std::fmt;
    use std::ops::Range;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use object_store::CopyOptions;
    use object_store::GetResult;
    use object_store::ListResult;
    use object_store::MultipartUpload;
    use object_store::PutMultipartOptions;
    use object_store::PutOptions;
    use object_store::PutPayload;
    use object_store::PutResult;
    use object_store::RenameOptions;
    use object_store::Result as ObjectStoreResult;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;

    use super::*;
    use crate::runtime::AbortHandle;
    use crate::runtime::AbortHandleRef;
    use crate::runtime::Executor;

    const TEST_DATA: &[u8] = b"object store test data";

    #[derive(Clone, Copy, Debug, Default)]
    enum ResponseMutation {
        #[default]
        None,
        Range,
        Metadata,
    }

    #[derive(Clone, Copy, Debug)]
    enum StreamBehavior {
        Error,
        Short,
        Pending,
    }

    #[derive(Debug)]
    struct RecordingStore<T> {
        inner: T,
        get_opts_calls: AtomicUsize,
        get_ranges_calls: AtomicUsize,
        head_calls: AtomicUsize,
        options: Mutex<Vec<GetOptions>>,
        mutation: ResponseMutation,
        stream_behaviors: Mutex<VecDeque<StreamBehavior>>,
        pending_stream_polls: Arc<AtomicUsize>,
    }

    impl<T> RecordingStore<T> {
        fn new(inner: T) -> Self {
            Self {
                inner,
                get_opts_calls: AtomicUsize::new(0),
                get_ranges_calls: AtomicUsize::new(0),
                head_calls: AtomicUsize::new(0),
                options: Mutex::new(Vec::new()),
                mutation: ResponseMutation::None,
                stream_behaviors: Mutex::new(VecDeque::new()),
                pending_stream_polls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_mutation(mut self, mutation: ResponseMutation) -> Self {
            self.mutation = mutation;
            self
        }

        fn with_stream_behaviors(
            self,
            behaviors: impl IntoIterator<Item = StreamBehavior>,
        ) -> Self {
            *self.stream_behaviors.lock() = behaviors.into_iter().collect();
            self
        }
    }

    impl<T: fmt::Display> fmt::Display for RecordingStore<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "RecordingStore({})", self.inner)
        }
    }

    #[async_trait]
    impl<T: ObjectStore> ObjectStore for RecordingStore<T> {
        async fn put_opts(
            &self,
            location: &ObjectPath,
            payload: PutPayload,
            options: PutOptions,
        ) -> ObjectStoreResult<PutResult> {
            self.inner.put_opts(location, payload, options).await
        }

        async fn put_multipart_opts(
            &self,
            location: &ObjectPath,
            options: PutMultipartOptions,
        ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, options).await
        }

        async fn get_opts(
            &self,
            location: &ObjectPath,
            options: GetOptions,
        ) -> ObjectStoreResult<GetResult> {
            if options.head {
                self.head_calls.fetch_add(1, Ordering::SeqCst);
            } else {
                self.get_opts_calls.fetch_add(1, Ordering::SeqCst);
            }
            self.options.lock().push(options.clone());
            let mut response = self.inner.get_opts(location, options).await?;
            match self.mutation {
                ResponseMutation::None => {}
                ResponseMutation::Range => {
                    response.range.end = response.range.end.saturating_add(1)
                }
                ResponseMutation::Metadata => {
                    response.meta.e_tag = Some("mismatched-etag".to_owned());
                }
            }
            match self.stream_behaviors.lock().pop_front() {
                None => {}
                Some(StreamBehavior::Error) => {
                    response.payload = GetResultPayload::Stream(
                        stream::once(async {
                            Err(object_store::Error::Generic {
                                store: "recording test store",
                                source: Box::new(io::Error::other("injected stream failure")),
                            })
                        })
                        .boxed(),
                    );
                }
                Some(StreamBehavior::Short) => {
                    response.payload = GetResultPayload::Stream(
                        stream::once(async { Ok(Bytes::from_static(b"short")) }).boxed(),
                    );
                }
                Some(StreamBehavior::Pending) => {
                    let pending_stream_polls = Arc::clone(&self.pending_stream_polls);
                    response.payload = GetResultPayload::Stream(
                        stream::once(async move {
                            pending_stream_polls.fetch_add(1, Ordering::SeqCst);
                            futures::future::pending::<ObjectStoreResult<Bytes>>().await
                        })
                        .boxed(),
                    );
                }
            }
            Ok(response)
        }

        async fn get_ranges(
            &self,
            location: &ObjectPath,
            ranges: &[Range<u64>],
        ) -> ObjectStoreResult<Vec<Bytes>> {
            self.get_ranges_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, ObjectStoreResult<ObjectPath>>,
        ) -> BoxStream<'static, ObjectStoreResult<ObjectPath>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&ObjectPath>,
        ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        fn list_with_offset(
            &self,
            prefix: Option<&ObjectPath>,
            offset: &ObjectPath,
        ) -> BoxStream<'static, ObjectStoreResult<ObjectMeta>> {
            self.inner.list_with_offset(prefix, offset)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&ObjectPath>,
        ) -> ObjectStoreResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &ObjectPath,
            to: &ObjectPath,
            options: CopyOptions,
        ) -> ObjectStoreResult<()> {
            self.inner.copy_opts(from, to, options).await
        }

        async fn rename_opts(
            &self,
            from: &ObjectPath,
            to: &ObjectPath,
            options: RenameOptions,
        ) -> ObjectStoreResult<()> {
            self.inner.rename_opts(from, to, options).await
        }
    }

    #[derive(Default)]
    struct CountingExecutor {
        spawn_count: AtomicUsize,
        spawn_io_count: AtomicUsize,
    }

    impl Executor for CountingExecutor {
        fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_io(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
            self.spawn_io_count.fetch_add(1, Ordering::SeqCst);
            TokioAbortHandle::new_handle(tokio::spawn(fut).abort_handle())
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::spawn(async move { task() }).abort_handle())
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::new_handle(tokio::task::spawn_blocking(task).abort_handle())
        }
    }

    struct TokioAbortHandle(tokio::task::AbortHandle);

    impl TokioAbortHandle {
        fn new_handle(handle: tokio::task::AbortHandle) -> AbortHandleRef {
            Box::new(Self(handle))
        }
    }

    impl AbortHandle for TokioAbortHandle {
        fn abort(self: Box<Self>) {
            self.0.abort();
        }
    }

    #[cfg(any(unix, windows))]
    async fn promoting_stream_reader(
        behaviors: impl IntoIterator<Item = StreamBehavior>,
    ) -> anyhow::Result<(
        PromotingObjectStoreReadAt,
        Arc<RecordingStore<InMemory>>,
        Arc<CountingExecutor>,
    )> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));
        let inner = InMemory::new();
        let path = ObjectPath::from("test.bin");
        inner.put(&path, PutPayload::from_static(TEST_DATA)).await?;
        let expected_meta = inner.head(&path).await?;
        let store = Arc::new(RecordingStore::new(inner).with_stream_behaviors(behaviors));
        let reader =
            ObjectStoreReadAt::new(Arc::clone(&store) as Arc<dyn ObjectStore>, path, handle)
                .into_file_payload_promoting(expected_meta, false)?;
        Ok((reader, store, executor))
    }

    #[tokio::test]
    async fn read_at_uses_spawn_io() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let buffer = reader.read_at(7, 5, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[tokio::test]
    async fn read_ranges_uses_one_io_task() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let path = ObjectPath::from("test.bin");
        store.put(&path, PutPayload::from_static(TEST_DATA)).await?;

        let reader = ObjectStoreReadAt::new(store, path, handle);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
            ReadAtRequest::new(18, 4, Alignment::new(1)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;

        assert_eq!(results.len(), 3);
        for (request, result) in results {
            let buffer = result?;
            let offset = usize::try_from(request.offset)?;
            assert_eq!(buffer.len(), request.length);
            assert_eq!(
                buffer.to_host().await.as_slice(),
                &TEST_DATA[offset..offset + request.length]
            );
        }
        assert_eq!(executor.spawn_io_count.load(Ordering::SeqCst), 1);
        assert_eq!(executor.spawn_count.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn unpromoted_local_file_preserves_legacy_get_options_and_bytes() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let directory = tempfile::tempdir()?;
        let path = ObjectPath::from("test.bin");
        std::fs::write(directory.path().join(path.as_ref()), TEST_DATA)?;
        let store = Arc::new(RecordingStore::new(LocalFileSystem::new_with_prefix(
            directory.path(),
        )?));
        let reader =
            ObjectStoreReadAt::new(Arc::clone(&store) as Arc<dyn ObjectStore>, path, handle);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
        ]);
        let mut results = reader
            .read_ranges(Arc::clone(&requests))
            .collect::<Vec<_>>()
            .await;
        results.sort_unstable_by_key(|(request, _)| request.offset);

        assert_eq!(results.len(), requests.len());
        for ((request, result), expected) in
            results.into_iter().zip([b"object".as_slice(), b"store"])
        {
            assert_eq!(result?.to_host().await.as_slice(), expected);
            assert_eq!(request.length, expected.len());
        }

        assert_eq!(store.get_opts_calls.load(Ordering::SeqCst), requests.len());
        assert_eq!(store.get_ranges_calls.load(Ordering::SeqCst), 0);
        let mut requested_ranges = Vec::new();
        for options in store.options.lock().iter() {
            anyhow::ensure!(options.if_match.is_none());
            anyhow::ensure!(options.if_none_match.is_none());
            anyhow::ensure!(options.if_modified_since.is_none());
            anyhow::ensure!(options.if_unmodified_since.is_none());
            anyhow::ensure!(options.version.is_none());
            anyhow::ensure!(!options.head);
            let Some(GetRange::Bounded(range)) = options.range.as_ref() else {
                anyhow::bail!("expected a bounded legacy range, got {:?}", options.range);
            };
            requested_ranges.push(range.clone());
        }
        requested_ranges.sort_unstable_by_key(|range| range.start);
        assert_eq!(requested_ranges, vec![0..6, 7..12]);

        Ok(())
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn promoting_size_uses_frozen_metadata_without_head() -> anyhow::Result<()> {
        let executor = Arc::new(CountingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let handle = Handle::new(Arc::downgrade(&runtime));

        let inner = InMemory::new();
        let path = ObjectPath::from("test.bin");
        inner.put(&path, PutPayload::from_static(TEST_DATA)).await?;
        let expected_meta = inner.head(&path).await?;
        let expected_size = expected_meta.size;
        let store = Arc::new(RecordingStore::new(inner));
        let reader = ObjectStoreReadAt::new(
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            path.clone(),
            handle,
        )
        .into_file_payload_promoting(expected_meta, false)?;
        store
            .put(
                &path,
                PutPayload::from_static(b"replacement with a different size"),
            )
            .await?;

        assert_eq!(reader.size().await?, expected_size);
        assert_eq!(store.head_calls.load(Ordering::SeqCst), 0);
        assert_eq!(store.get_opts_calls.load(Ordering::SeqCst), 0);
        assert_ne!(store.head(&path).await?.size, expected_size);
        assert_eq!(store.head_calls.load(Ordering::SeqCst), 1);

        Ok(())
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn stream_mismatch_does_not_disable_promotion() -> anyhow::Result<()> {
        for mutation in [ResponseMutation::Range, ResponseMutation::Metadata] {
            let executor = Arc::new(CountingExecutor::default());
            let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
            let handle = Handle::new(Arc::downgrade(&runtime));

            let inner = InMemory::new();
            let path = ObjectPath::from("test.bin");
            inner.put(&path, PutPayload::from_static(TEST_DATA)).await?;
            let expected_meta = inner.head(&path).await?;
            let store = Arc::new(RecordingStore::new(inner).with_mutation(mutation));
            let reader =
                ObjectStoreReadAt::new(Arc::clone(&store) as Arc<dyn ObjectStore>, path, handle)
                    .into_file_payload_promoting(expected_meta, false)?;

            anyhow::ensure!(reader.read_at(0, 6, Alignment::new(1)).await.is_err());
            assert!(matches!(
                &*reader.promotion.state.lock(),
                FilePayloadPromotionState::Eligible
            ));
        }

        Ok(())
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn matching_stream_installs_non_file_only_after_success() -> anyhow::Result<()> {
        let (reader, store, _executor) = promoting_stream_reader([]).await?;

        assert!(matches!(
            &*reader.promotion.state.lock(),
            FilePayloadPromotionState::Eligible
        ));
        let buffer = reader.read_at(0, 6, Alignment::new(1)).await?;
        assert_eq!(buffer.to_host().await.as_slice(), b"object");
        assert!(matches!(
            &*reader.promotion.state.lock(),
            FilePayloadPromotionState::NonFile
        ));
        assert_eq!(store.get_opts_calls.load(Ordering::SeqCst), 1);

        Ok(())
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn failed_or_short_stream_remains_eligible_and_retries() -> anyhow::Result<()> {
        for behavior in [StreamBehavior::Error, StreamBehavior::Short] {
            let (reader, store, _executor) = promoting_stream_reader([behavior]).await?;

            assert!(reader.read_at(0, 6, Alignment::new(1)).await.is_err());
            assert!(matches!(
                &*reader.promotion.state.lock(),
                FilePayloadPromotionState::Eligible
            ));

            let buffer = reader.read_at(0, 6, Alignment::new(1)).await?;
            assert_eq!(buffer.to_host().await.as_slice(), b"object");
            assert!(matches!(
                &*reader.promotion.state.lock(),
                FilePayloadPromotionState::NonFile
            ));
            assert_eq!(store.get_opts_calls.load(Ordering::SeqCst), 2);
        }

        Ok(())
    }

    #[cfg(any(unix, windows))]
    #[tokio::test]
    async fn cancelled_stream_remains_eligible_and_retries() -> anyhow::Result<()> {
        let (reader, store, _executor) = promoting_stream_reader([StreamBehavior::Pending]).await?;

        let task = tokio::spawn(reader.read_at(0, 6, Alignment::new(1)));
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while store.pending_stream_polls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await?;
        task.abort();
        drop(task.await);

        assert!(matches!(
            &*reader.promotion.state.lock(),
            FilePayloadPromotionState::Eligible
        ));
        let buffer = reader.read_at(0, 6, Alignment::new(1)).await?;
        assert_eq!(buffer.to_host().await.as_slice(), b"object");
        assert!(matches!(
            &*reader.promotion.state.lock(),
            FilePayloadPromotionState::NonFile
        ));
        assert_eq!(store.get_opts_calls.load(Ordering::SeqCst), 2);

        Ok(())
    }
}
