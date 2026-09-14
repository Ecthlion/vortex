// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diagnostic-only object-store reads for attributing local positional I/O.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use futures::FutureExt;
use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use tracing::Instrument;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::diagnostic_timestamp_ns;
use crate::object_store::DEFAULT_CONCURRENCY;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

static SCAN_DIAGNOSTICS_ENABLED: OnceLock<bool> = OnceLock::new();
static NEXT_READER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WORKER_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static DIAGNOSTIC_WORKER_ID: u64 = NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
}

pub(super) fn scan_diagnostics_enabled() -> bool {
    *SCAN_DIAGNOSTICS_ENABLED.get_or_init(|| {
        std::env::var_os("VORTEX_SCAN_DIAGNOSTICS").is_some_and(|value| value != "0")
    })
}

/// A separate implementation keeps the ordinary [`crate::object_store::ObjectStoreReadAt`] path
/// free of diagnostic fields and branches.
pub(super) struct DiagnosticObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: HostAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    diagnostics: Arc<DiagnosticSummary>,
}

impl DiagnosticObjectStoreReadAt {
    pub(super) fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self::new_with_allocator(store, path, handle, Arc::new(DefaultHostAllocator))
    }

    fn new_with_allocator(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        let reader_id = NEXT_READER_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            store,
            path,
            uri: Arc::clone(&uri),
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
            diagnostics: Arc::new(DiagnosticSummary::new(reader_id, Arc::clone(&uri))),
        }
    }

    fn child(
        &self,
        child_index: usize,
        request: ReadAtRequest,
        parent: tracing::Span,
    ) -> ChildTerminal {
        ChildTerminal::new(
            Arc::clone(&self.diagnostics),
            child_index,
            Arc::clone(&self.uri),
            request,
            parent,
        )
    }
}

impl VortexReadAt for DiagnosticObjectStoreReadAt {
    fn diagnostic_instance_id(&self) -> Option<u64> {
        Some(self.diagnostics.reader_id)
    }

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
                .map(|metadata| metadata.size)
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
        let request = ReadAtRequest::new(offset, length, alignment);
        let direct_read_id = self
            .diagnostics
            .direct_read_calls
            .fetch_add(1, Ordering::Relaxed);
        let span = tracing::trace_span!(
            target: "vortex_io::object_store_direct_read",
            "object_store_direct_read",
            read_at_id = self.diagnostics.reader_id,
            direct_read_id,
        );
        let terminal = self.child(0, request, span.clone());
        handle
            .spawn_io(
                async move {
                    read_diagnostic_object_store_range(
                        store, path, io_handle, allocator, request, terminal,
                    )
                    .await
                }
                .instrument(span),
            )
            .boxed()
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        self.diagnostics
            .read_ranges_calls
            .fetch_add(1, Ordering::Relaxed);
        if requests.is_empty() {
            return stream::empty().boxed();
        }

        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        let concurrency = self.concurrency.max(1);
        let parent = tracing::Span::current();
        let diagnostics = Arc::clone(&self.diagnostics);
        let uri = Arc::clone(&self.uri);

        async_stream::stream! {
            let (mut send, mut recv) = mpsc::channel(concurrency);
            let io_handle = handle.clone();
            let task = handle.spawn_io(async move {
                let reads = stream::iter(0..requests.len()).map(move |child_index| {
                    let request = requests[child_index];
                    let store = Arc::clone(&store);
                    let path = path.clone();
                    let io_handle = io_handle.clone();
                    let allocator = Arc::clone(&allocator);
                    let diagnostics = Arc::clone(&diagnostics);
                    let uri = Arc::clone(&uri);
                    let parent = parent.clone();
                    async move {
                        // `buffer_unordered` constructs this terminal only after admitting and
                        // polling this child. Merely constructing or dropping the outer stream
                        // therefore does not report physical I/O that never existed.
                        let terminal =
                            ChildTerminal::new(diagnostics, child_index, uri, request, parent);
                        let result = read_diagnostic_object_store_range(
                            store, path, io_handle, allocator, request, terminal,
                        )
                        .await;
                        (request, result)
                    }
                });

                let mut reads = reads.buffer_unordered(concurrency);
                while let Some(result) = reads.next().await {
                    if send.send(result).await.is_err() {
                        break;
                    }
                }
            });
            while let Some(result) = recv.next().await {
                yield result;
            }
            task.await;
        }
        .boxed()
    }
}

async fn read_diagnostic_object_store_range(
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    io_handle: Handle,
    allocator: HostAllocatorRef,
    request: ReadAtRequest,
    mut terminal: ChildTerminal,
) -> VortexResult<BufferHandle> {
    let _consumer = terminal.consumer_guard();
    let ReadAtRequest {
        offset,
        length,
        alignment,
    } = request;
    let end = offset
        .checked_add(u64::try_from(length)?)
        .ok_or_else(|| vortex_err!("positional read range overflow"));
    let end = match end {
        Ok(end) => end,
        Err(error) => {
            terminal.error("range", &error);
            return Err(error);
        }
    };
    let range = offset..end;
    let mut buffer = match allocator.allocate(length, alignment) {
        Ok(buffer) => buffer,
        Err(error) => {
            terminal.error("allocate", &error);
            return Err(error);
        }
    };

    let get_opts_started = Instant::now();
    let response = store
        .get_opts(
            &path,
            GetOptions {
                range: Some(GetRange::Bounded(range.clone())),
                ..Default::default()
            },
        )
        .await;
    terminal.get_opts = Some(get_opts_started.elapsed());
    terminal.t_get_opts_done_ns = Some(diagnostic_timestamp_ns());
    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let error = VortexError::from(error);
            terminal.error("get_opts", &error);
            return Err(error);
        }
    };

    let buffer = match response.payload {
        #[cfg(not(target_arch = "wasm32"))]
        GetResultPayload::File(file, _) => {
            terminal.payload = Payload::File;
            let enqueued = Instant::now();
            terminal.t_blocking_enqueued_ns = Some(diagnostic_timestamp_ns());
            return io_handle
                .spawn_blocking(move || {
                    let started = Instant::now();
                    terminal.mark_blocking_started(enqueued);
                    let result =
                        read_exact_at(&file, buffer.as_mut_slice(), range.start).map(|()| buffer);
                    terminal.service = Some(started.elapsed());
                    match result {
                        Ok(buffer) => {
                            let buffer = BufferHandle::new_host(buffer.freeze());
                            terminal.success(buffer.len());
                            Ok(buffer)
                        }
                        Err(error) => {
                            let error = VortexError::from(error);
                            terminal.error("pread", &error);
                            Err(error)
                        }
                    }
                })
                .await;
        }
        #[cfg(target_arch = "wasm32")]
        GetResultPayload::File(..) => unreachable!("File payload not supported on wasm32"),
        GetResultPayload::Stream(mut byte_stream) => {
            terminal.payload = Payload::Stream;
            let service_started = Instant::now();
            let mut written = 0usize;
            while let Some(bytes) = byte_stream.next().await {
                let bytes = match bytes {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        terminal.service = Some(service_started.elapsed());
                        let error = VortexError::from(error);
                        terminal.error("stream", &error);
                        return Err(error);
                    }
                };
                terminal
                    .first_byte
                    .get_or_insert_with(|| service_started.elapsed());
                terminal
                    .t_first_byte_ns
                    .get_or_insert_with(diagnostic_timestamp_ns);
                let Some(next_written) = written.checked_add(bytes.len()) else {
                    terminal.service = Some(service_started.elapsed());
                    let error = vortex_err!("object store stream byte count overflow");
                    terminal.error("stream", &error);
                    return Err(error);
                };
                if next_written > length {
                    terminal.service = Some(service_started.elapsed());
                    let error = vortex_err!(
                        "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                        next_written,
                        length,
                        range
                    );
                    terminal.error("stream", &error);
                    return Err(error);
                }
                buffer.as_mut_slice()[written..next_written].copy_from_slice(&bytes);
                written = next_written;
                terminal.bytes = written;
            }
            terminal.service = Some(service_started.elapsed());

            if written != length {
                let error = vortex_err!(
                    "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                    written,
                    length,
                    range
                );
                terminal.error("stream", &error);
                return Err(error);
            }
            buffer
        }
    };

    let buffer = BufferHandle::new_host(buffer.freeze());
    terminal.success(buffer.len());
    Ok(buffer)
}

#[derive(Clone, Copy)]
enum Payload {
    None,
    File,
    Stream,
}

impl Payload {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::File => "file",
            Self::Stream => "stream",
        }
    }
}

struct ChildTerminal {
    summary: Arc<DiagnosticSummary>,
    child_index: usize,
    path: Arc<str>,
    request: ReadAtRequest,
    parent: tracing::Span,
    lifecycle: Arc<ChildLifecycle>,
    payload: Payload,
    t_begin_ns: u64,
    t_get_opts_done_ns: Option<u64>,
    t_blocking_enqueued_ns: Option<u64>,
    t_blocking_start_ns: Option<u64>,
    t_first_byte_ns: Option<u64>,
    get_opts: Option<Duration>,
    blocking_queue: Option<Duration>,
    service: Option<Duration>,
    first_byte: Option<Duration>,
    blocking_worker_id: Option<u64>,
    blocking_worker_name: Option<String>,
    bytes: usize,
    finished: bool,
}

impl ChildTerminal {
    fn new(
        summary: Arc<DiagnosticSummary>,
        child_index: usize,
        path: Arc<str>,
        request: ReadAtRequest,
        parent: tracing::Span,
    ) -> Self {
        summary.children.fetch_add(1, Ordering::Relaxed);
        summary.requested_bytes.fetch_add(
            u64::try_from(request.length).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Self {
            summary,
            child_index,
            path,
            request,
            parent,
            lifecycle: Arc::new(ChildLifecycle::default()),
            payload: Payload::None,
            t_begin_ns: diagnostic_timestamp_ns(),
            t_get_opts_done_ns: None,
            t_blocking_enqueued_ns: None,
            t_blocking_start_ns: None,
            t_first_byte_ns: None,
            get_opts: None,
            blocking_queue: None,
            service: None,
            first_byte: None,
            blocking_worker_id: None,
            blocking_worker_name: None,
            bytes: 0,
            finished: false,
        }
    }

    fn consumer_guard(&self) -> ConsumerGuard {
        ConsumerGuard {
            summary: Arc::clone(&self.summary),
            lifecycle: Arc::clone(&self.lifecycle),
        }
    }

    fn mark_blocking_started(&mut self, enqueued: Instant) {
        self.blocking_queue = Some(enqueued.elapsed());
        self.t_blocking_start_ns = Some(diagnostic_timestamp_ns());
        let worker = thread::current();
        self.blocking_worker_id = Some(DIAGNOSTIC_WORKER_ID.with(|id| *id));
        self.blocking_worker_name = worker.name().map(str::to_owned);
    }

    fn success(&mut self, bytes: usize) {
        self.summary.successes.fetch_add(1, Ordering::Relaxed);
        self.bytes = bytes;
        self.finish("success", "", "");
    }

    fn error(&mut self, stage: &'static str, error: &VortexError) {
        self.summary.errors.fetch_add(1, Ordering::Relaxed);
        self.finish("error", stage, &error.to_string());
    }

    fn finish(&mut self, outcome: &'static str, error_stage: &'static str, error: &str) {
        let t_finish_ns = diagnostic_timestamp_ns();
        let consumer_cancelled = self.lifecycle.finish();
        self.record_durations();
        self.summary.terminals.fetch_add(1, Ordering::Relaxed);
        self.summary.completed_bytes.fetch_add(
            u64::try_from(self.bytes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        match self.payload {
            Payload::None => &self.summary.no_payloads,
            Payload::File => &self.summary.file_payloads,
            Payload::Stream => &self.summary.stream_payloads,
        }
        .fetch_add(1, Ordering::Relaxed);
        let get_opts_ns = duration_ns(self.get_opts);
        let blocking_queue_ns = duration_ns(self.blocking_queue);
        let service_ns = duration_ns(self.service);
        let first_byte_ns = duration_ns(self.first_byte);
        tracing::dispatcher::with_default(&self.summary.dispatch, || {
            tracing::trace!(
                target: "vortex_io::object_store_child",
                parent: &self.parent,
                read_at_id = self.summary.reader_id,
                child_index = self.child_index,
                path = %self.path,
                offset = self.request.offset,
                length = self.request.length,
                alignment = *self.request.alignment,
                payload = self.payload.as_str(),
                outcome,
                consumer_cancelled,
                bytes = self.bytes,
                t_begin_ns = self.t_begin_ns,
                t_get_opts_done_ns = self.t_get_opts_done_ns.unwrap_or_default(),
                t_blocking_enqueued_ns = self.t_blocking_enqueued_ns.unwrap_or_default(),
                t_blocking_start_ns = self.t_blocking_start_ns.unwrap_or_default(),
                t_first_byte_ns = self.t_first_byte_ns.unwrap_or_default(),
                t_finish_ns,
                get_opts_ns,
                blocking_queue_ns,
                service_ns,
                first_byte_ns,
                has_get_opts = self.get_opts.is_some(),
                has_get_opts_done = self.t_get_opts_done_ns.is_some(),
                has_blocking_enqueued = self.t_blocking_enqueued_ns.is_some(),
                has_blocking_start = self.t_blocking_start_ns.is_some(),
                has_blocking_queue = self.blocking_queue.is_some(),
                has_service = self.service.is_some(),
                has_first_byte = self.first_byte.is_some(),
                blocking_worker_id = self.blocking_worker_id.unwrap_or_default(),
                has_blocking_worker_id = self.blocking_worker_id.is_some(),
                blocking_worker_name = self.blocking_worker_name.as_deref().unwrap_or(""),
                error_stage,
                error,
                "object-store physical child completed"
            );
        });
        self.finished = true;
    }

    fn record_durations(&self) {
        self.summary.record_duration(
            &self.summary.get_opts_observations,
            &self.summary.get_opts_ns,
            self.get_opts,
        );
        self.summary.record_duration(
            &self.summary.blocking_queue_observations,
            &self.summary.blocking_queue_ns,
            self.blocking_queue,
        );
        self.summary.record_duration(
            &self.summary.service_observations,
            &self.summary.service_ns,
            self.service,
        );
    }
}

impl Drop for ChildTerminal {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.summary
            .physical_cancellations
            .fetch_add(1, Ordering::Relaxed);
        self.finish("cancel", "", "");
    }
}

const CHILD_ACTIVE: u8 = 0;
const CHILD_CONSUMER_CANCELLED: u8 = 1;
const CHILD_FINISHED: u8 = 2;
const CHILD_FINISHED_AFTER_CONSUMER_CANCEL: u8 = 3;

#[derive(Default)]
struct ChildLifecycle {
    state: AtomicU8,
}

impl ChildLifecycle {
    fn cancel_consumer(&self) -> bool {
        self.state
            .compare_exchange(
                CHILD_ACTIVE,
                CHILD_CONSUMER_CANCELLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    fn finish(&self) -> bool {
        loop {
            match self.state.load(Ordering::Acquire) {
                CHILD_ACTIVE => {
                    if self
                        .state
                        .compare_exchange(
                            CHILD_ACTIVE,
                            CHILD_FINISHED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return false;
                    }
                }
                CHILD_CONSUMER_CANCELLED => {
                    if self
                        .state
                        .compare_exchange(
                            CHILD_CONSUMER_CANCELLED,
                            CHILD_FINISHED_AFTER_CONSUMER_CANCEL,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return true;
                    }
                }
                CHILD_FINISHED => return false,
                CHILD_FINISHED_AFTER_CONSUMER_CANCEL => return true,
                _ => unreachable!("invalid child diagnostic lifecycle state"),
            }
        }
    }
}

struct ConsumerGuard {
    summary: Arc<DiagnosticSummary>,
    lifecycle: Arc<ChildLifecycle>,
}

impl Drop for ConsumerGuard {
    fn drop(&mut self) {
        if self.lifecycle.cancel_consumer() {
            self.summary
                .consumer_cancellations
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct DiagnosticSummary {
    reader_id: u64,
    path: Arc<str>,
    dispatch: tracing::Dispatch,
    direct_read_calls: AtomicU64,
    read_ranges_calls: AtomicU64,
    children: AtomicU64,
    terminals: AtomicU64,
    successes: AtomicU64,
    errors: AtomicU64,
    physical_cancellations: AtomicU64,
    consumer_cancellations: AtomicU64,
    requested_bytes: AtomicU64,
    completed_bytes: AtomicU64,
    file_payloads: AtomicU64,
    stream_payloads: AtomicU64,
    no_payloads: AtomicU64,
    get_opts_observations: AtomicU64,
    get_opts_ns: AtomicU64,
    blocking_queue_observations: AtomicU64,
    blocking_queue_ns: AtomicU64,
    service_observations: AtomicU64,
    service_ns: AtomicU64,
}

impl DiagnosticSummary {
    fn new(reader_id: u64, path: Arc<str>) -> Self {
        Self {
            reader_id,
            path,
            dispatch: tracing::dispatcher::get_default(Clone::clone),
            direct_read_calls: AtomicU64::new(0),
            read_ranges_calls: AtomicU64::new(0),
            children: AtomicU64::new(0),
            terminals: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            physical_cancellations: AtomicU64::new(0),
            consumer_cancellations: AtomicU64::new(0),
            requested_bytes: AtomicU64::new(0),
            completed_bytes: AtomicU64::new(0),
            file_payloads: AtomicU64::new(0),
            stream_payloads: AtomicU64::new(0),
            no_payloads: AtomicU64::new(0),
            get_opts_observations: AtomicU64::new(0),
            get_opts_ns: AtomicU64::new(0),
            blocking_queue_observations: AtomicU64::new(0),
            blocking_queue_ns: AtomicU64::new(0),
            service_observations: AtomicU64::new(0),
            service_ns: AtomicU64::new(0),
        }
    }

    fn record_duration(
        &self,
        observations: &AtomicU64,
        counter: &AtomicU64,
        duration: Option<Duration>,
    ) {
        if let Some(duration) = duration {
            observations.fetch_add(1, Ordering::Relaxed);
            counter.fetch_add(nanoseconds(duration), Ordering::Relaxed);
        }
    }
}

impl Drop for DiagnosticSummary {
    fn drop(&mut self) {
        let direct_read_calls = self.direct_read_calls.load(Ordering::Relaxed);
        let read_ranges_calls = self.read_ranges_calls.load(Ordering::Relaxed);
        let dispatch = self.dispatch.clone();
        tracing::dispatcher::with_default(&dispatch, || {
            tracing::debug!(
                target: "vortex_io::object_store_summary",
                scope = "reader",
                read_at_id = self.reader_id,
                path = %self.path,
                read_calls = direct_read_calls.saturating_add(read_ranges_calls),
                direct_read_calls,
                read_ranges_calls,
                children = self.children.load(Ordering::Relaxed),
                terminals = self.terminals.load(Ordering::Relaxed),
                successes = self.successes.load(Ordering::Relaxed),
                errors = self.errors.load(Ordering::Relaxed),
                physical_cancellations = self.physical_cancellations.load(Ordering::Relaxed),
                consumer_cancellations = self.consumer_cancellations.load(Ordering::Relaxed),
                requested_bytes = self.requested_bytes.load(Ordering::Relaxed),
                completed_bytes = self.completed_bytes.load(Ordering::Relaxed),
                file_payloads = self.file_payloads.load(Ordering::Relaxed),
                stream_payloads = self.stream_payloads.load(Ordering::Relaxed),
                no_payloads = self.no_payloads.load(Ordering::Relaxed),
                get_opts_observations = self.get_opts_observations.load(Ordering::Relaxed),
                get_opts_ns = self.get_opts_ns.load(Ordering::Relaxed),
                blocking_queue_observations = self.blocking_queue_observations.load(Ordering::Relaxed),
                blocking_queue_ns = self.blocking_queue_ns.load(Ordering::Relaxed),
                service_observations = self.service_observations.load(Ordering::Relaxed),
                service_ns = self.service_ns.load(Ordering::Relaxed),
                "object-store diagnostic summary"
            );
        });
    }
}

fn duration_ns(duration: Option<Duration>) -> u64 {
    duration.map_or(0, nanoseconds)
}

fn nanoseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[cfg(feature = "tokio")]
mod tests {
    use std::collections::BTreeMap;
    use std::fmt;
    use std::io;
    use std::ops::Range;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc as std_mpsc;

    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use object_store::CopyOptions;
    use object_store::GetResult;
    use object_store::ListResult;
    use object_store::MultipartUpload;
    use object_store::ObjectMeta;
    use object_store::PutMultipartOptions;
    use object_store::PutOptions;
    use object_store::PutPayload;
    use object_store::PutResult;
    use object_store::RenameOptions;
    use object_store::Result as ObjectStoreResult;
    use object_store::local::LocalFileSystem;
    use object_store::memory::InMemory;
    use tracing::Event;
    use tracing::Subscriber;
    use tracing::field::Field;
    use tracing::field::Visit;
    use tracing::span::Attributes;
    use tracing_subscriber::Layer;
    use tracing_subscriber::filter::Targets;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::registry::LookupSpan;

    use super::*;
    use crate::runtime::AbortHandle;
    use crate::runtime::AbortHandleRef;
    use crate::runtime::Executor;
    use crate::runtime::tokio::TokioRuntime;

    const TEST_DATA: &[u8] = b"object store test data";

    #[derive(Debug)]
    struct CapturedEvent {
        fields: BTreeMap<String, String>,
        parent_name: String,
        parent_fields: BTreeMap<String, String>,
    }

    #[derive(Default)]
    struct TraceCapture {
        events: parking_lot::Mutex<Vec<CapturedEvent>>,
        summaries: parking_lot::Mutex<Vec<CapturedEvent>>,
    }

    struct CaptureLayer {
        capture: Arc<TraceCapture>,
    }

    #[derive(Default)]
    struct CapturedFields(BTreeMap<String, String>);

    #[derive(Default)]
    struct FieldVisitor(BTreeMap<String, String>);

    impl Visit for FieldVisitor {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.insert(field.name().to_owned(), value.to_string());
        }

        fn record_i64(&mut self, field: &Field, value: i64) {
            self.0.insert(field.name().to_owned(), value.to_string());
        }

        fn record_bool(&mut self, field: &Field, value: bool) {
            self.0.insert(field.name().to_owned(), value.to_string());
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_owned(), value.to_owned());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0.insert(field.name().to_owned(), format!("{value:?}"));
        }
    }

    impl<S> Layer<S> for CaptureLayer
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            attributes: &Attributes<'_>,
            id: &tracing::span::Id,
            ctx: Context<'_, S>,
        ) {
            let mut visitor = FieldVisitor::default();
            attributes.record(&mut visitor);
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(CapturedFields(visitor.0));
            }
        }

        fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
            let target = event.metadata().target();
            if target != "vortex_io::object_store_child"
                && target != "vortex_io::object_store_summary"
            {
                return;
            }
            let mut visitor = FieldVisitor::default();
            event.record(&mut visitor);
            let parent = event
                .parent()
                .and_then(|id| ctx.span(id))
                .or_else(|| ctx.lookup_current());
            let (parent_name, parent_fields) = parent.map_or_else(
                || (String::new(), BTreeMap::new()),
                |span| {
                    let fields = span
                        .extensions()
                        .get::<CapturedFields>()
                        .map_or_else(BTreeMap::new, |fields| fields.0.clone());
                    (span.name().to_owned(), fields)
                },
            );
            let captured = CapturedEvent {
                fields: visitor.0,
                parent_name,
                parent_fields,
            };
            if target == "vortex_io::object_store_child" {
                self.capture.events.lock().push(captured);
            } else {
                self.capture.summaries.lock().push(captured);
            }
        }
    }

    fn capture_traces() -> (Arc<TraceCapture>, tracing::dispatcher::DefaultGuard) {
        let capture = Arc::new(TraceCapture::default());
        let subscriber = tracing_subscriber::registry().with(CaptureLayer {
            capture: Arc::clone(&capture),
        });
        let guard = tracing::subscriber::set_default(subscriber);
        (capture, guard)
    }

    fn capture_runbook_traces() -> (Arc<TraceCapture>, tracing::dispatcher::DefaultGuard) {
        let capture = Arc::new(TraceCapture::default());
        let targets = Targets::new()
            .with_default(tracing::level_filters::LevelFilter::OFF)
            .with_target(
                "vortex_file::read_ranges",
                tracing::level_filters::LevelFilter::TRACE,
            )
            .with_target(
                "vortex_io::object_store_child",
                tracing::level_filters::LevelFilter::TRACE,
            )
            .with_target(
                "vortex_io::object_store_summary",
                tracing::level_filters::LevelFilter::DEBUG,
            );
        let subscriber = tracing_subscriber::registry().with(
            CaptureLayer {
                capture: Arc::clone(&capture),
            }
            .with_filter(targets),
        );
        let guard = tracing::subscriber::set_default(subscriber);
        (capture, guard)
    }

    fn assert_one_terminal(capture: &TraceCapture, outcome: &str) {
        let events = capture.events.lock();
        assert_eq!(events.len(), 1, "expected exactly one physical terminal");
        assert_eq!(events[0].fields["outcome"], outcome);
    }

    #[derive(Clone, Copy, Debug)]
    enum StreamBehavior {
        Normal,
        Error,
        Short,
        Pending,
    }

    #[derive(Debug)]
    struct BehaviorStore {
        inner: InMemory,
        behavior: StreamBehavior,
        pending_polls: Arc<AtomicUsize>,
    }

    impl fmt::Display for BehaviorStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "BehaviorStore")
        }
    }

    #[async_trait]
    impl ObjectStore for BehaviorStore {
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
            let mut result = self.inner.get_opts(location, options).await?;
            result.payload = match self.behavior {
                StreamBehavior::Normal => result.payload,
                StreamBehavior::Error => GetResultPayload::Stream(
                    stream::once(async {
                        Err(object_store::Error::Generic {
                            store: "diagnostic test store",
                            source: Box::new(io::Error::other("injected stream failure")),
                        })
                    })
                    .boxed(),
                ),
                StreamBehavior::Short => GetResultPayload::Stream(
                    stream::once(async { Ok(Bytes::from_static(b"short")) }).boxed(),
                ),
                StreamBehavior::Pending => {
                    let polls = Arc::clone(&self.pending_polls);
                    GetResultPayload::Stream(
                        stream::once(async move {
                            polls.fetch_add(1, Ordering::SeqCst);
                            futures::future::pending::<ObjectStoreResult<Bytes>>().await
                        })
                        .boxed(),
                    )
                }
            };
            Ok(result)
        }

        async fn get_ranges(
            &self,
            location: &ObjectPath,
            ranges: &[Range<u64>],
        ) -> ObjectStoreResult<Vec<Bytes>> {
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

    async fn stream_reader(
        behavior: StreamBehavior,
    ) -> anyhow::Result<(DiagnosticObjectStoreReadAt, Arc<AtomicUsize>)> {
        let inner = InMemory::new();
        let path = ObjectPath::from("test.bin");
        inner.put(&path, PutPayload::from_static(TEST_DATA)).await?;
        let pending_polls = Arc::new(AtomicUsize::new(0));
        let store = Arc::new(BehaviorStore {
            inner,
            behavior,
            pending_polls: Arc::clone(&pending_polls),
        });
        Ok((
            DiagnosticObjectStoreReadAt::new(
                store as Arc<dyn ObjectStore>,
                path,
                TokioRuntime::current(),
            ),
            pending_polls,
        ))
    }

    fn assert_summary_consistent(summary: &DiagnosticSummary) {
        let terminals = summary.terminals.load(Ordering::SeqCst);
        assert_eq!(summary.children.load(Ordering::SeqCst), terminals);
        assert_eq!(
            summary.successes.load(Ordering::SeqCst)
                + summary.errors.load(Ordering::SeqCst)
                + summary.physical_cancellations.load(Ordering::SeqCst),
            terminals
        );
        assert_eq!(
            summary.file_payloads.load(Ordering::SeqCst)
                + summary.stream_payloads.load(Ordering::SeqCst)
                + summary.no_payloads.load(Ordering::SeqCst),
            terminals
        );
    }

    #[tokio::test]
    async fn stream_success_has_one_terminal() -> anyhow::Result<()> {
        let (reader, _) = stream_reader(StreamBehavior::Normal).await?;
        let summary = Arc::clone(&reader.diagnostics);
        let buffer = reader.read_at(7, 5, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"store");
        assert_eq!(summary.successes.load(Ordering::SeqCst), 1);
        assert_eq!(summary.stream_payloads.load(Ordering::SeqCst), 1);
        assert_eq!(summary.completed_bytes.load(Ordering::SeqCst), 5);
        assert_summary_consistent(&summary);
        Ok(())
    }

    #[tokio::test]
    async fn read_ranges_has_one_terminal_per_child() -> anyhow::Result<()> {
        let (reader, _) = stream_reader(StreamBehavior::Normal).await?;
        let summary = Arc::clone(&reader.diagnostics);
        let requests: Arc<[ReadAtRequest]> = Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
            ReadAtRequest::new(18, 4, Alignment::new(1)),
        ]);
        let results = reader.read_ranges(requests).collect::<Vec<_>>().await;

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(_, result)| result.is_ok()));
        assert_eq!(summary.read_ranges_calls.load(Ordering::SeqCst), 1);
        assert_eq!(summary.children.load(Ordering::SeqCst), 3);
        assert_summary_consistent(&summary);
        Ok(())
    }

    #[tokio::test]
    async fn never_polled_read_ranges_has_no_physical_children() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let (reader, _) = stream_reader(StreamBehavior::Pending).await?;
        let summary = Arc::clone(&reader.diagnostics);
        let reads = reader.read_ranges(Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
        ]));

        drop(reads);

        assert_eq!(summary.read_ranges_calls.load(Ordering::SeqCst), 1);
        assert_eq!(summary.children.load(Ordering::SeqCst), 0);
        assert_eq!(summary.terminals.load(Ordering::SeqCst), 0);
        assert_eq!(summary.requested_bytes.load(Ordering::SeqCst), 0);
        assert_eq!(summary.physical_cancellations.load(Ordering::SeqCst), 0);
        assert_eq!(summary.consumer_cancellations.load(Ordering::SeqCst), 0);
        assert!(capture.events.lock().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn partially_admitted_read_ranges_cancels_only_polled_children() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let (mut reader, pending_polls) = stream_reader(StreamBehavior::Pending).await?;
        reader.concurrency = 1;
        let summary = Arc::clone(&reader.diagnostics);
        let mut reads = reader.read_ranges(Arc::from([
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            ReadAtRequest::new(7, 5, Alignment::new(1)),
            ReadAtRequest::new(18, 4, Alignment::new(1)),
        ]));

        assert!(reads.next().now_or_never().is_none());
        for _ in 0..100 {
            if pending_polls.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(pending_polls.load(Ordering::SeqCst), 1);
        assert_eq!(summary.children.load(Ordering::SeqCst), 1);
        assert_eq!(summary.terminals.load(Ordering::SeqCst), 0);
        assert_eq!(summary.requested_bytes.load(Ordering::SeqCst), 6);

        drop(reads);
        for _ in 0..100 {
            if summary.terminals.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }

        assert_eq!(summary.children.load(Ordering::SeqCst), 1);
        assert_eq!(summary.physical_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(summary.consumer_cancellations.load(Ordering::SeqCst), 1);
        assert_summary_consistent(&summary);
        let events = capture.events.lock();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].fields["child_index"], "0");
        assert_eq!(events[0].fields["outcome"], "cancel");
        Ok(())
    }

    #[tokio::test]
    async fn repeated_ranges_have_one_terminal_with_authoritative_parent_keys() -> anyhow::Result<()>
    {
        let (capture, _subscriber) = capture_traces();
        let (reader, _) = stream_reader(StreamBehavior::Normal).await?;
        let read_at_id = reader
            .diagnostic_instance_id()
            .ok_or_else(|| anyhow::anyhow!("diagnostic reader has no identity"))?;
        let parent = tracing::trace_span!(
            target: "vortex_file::read_ranges",
            "read_ranges",
            source_id = 41_u64,
            read_at_id,
            has_read_at_id = true,
            call_id = 7_u64,
        );
        let request = ReadAtRequest::new(7, 5, Alignment::new(1));
        let reads = {
            let _entered = parent.enter();
            reader.read_ranges(Arc::from([request, request]))
        };
        let results = reads.collect::<Vec<_>>().await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, result)| result.is_ok()));

        let events = capture.events.lock();
        assert_eq!(events.len(), 2, "one terminal event per repeated child");
        let mut child_indexes = events
            .iter()
            .map(|event| event.fields["child_index"].clone())
            .collect::<Vec<_>>();
        child_indexes.sort();
        assert_eq!(child_indexes, ["0", "1"]);
        for event in events.iter() {
            assert_eq!(event.parent_name, "read_ranges");
            assert_eq!(event.parent_fields["source_id"], "41");
            assert_eq!(event.parent_fields["read_at_id"], read_at_id.to_string());
            assert_eq!(event.parent_fields["call_id"], "7");
            assert_eq!(event.fields["read_at_id"], read_at_id.to_string());
            assert_eq!(event.fields["offset"], "7");
            assert_eq!(event.fields["length"], "5");
            assert_eq!(event.fields["outcome"], "success");
            assert_eq!(event.fields["consumer_cancelled"], "false");
            assert!(
                !event.fields.contains_key("call_id"),
                "the child must inherit, not invent, the physical call identity"
            );
            let begin = event.fields["t_begin_ns"].parse::<u64>()?;
            let get_opts_done = event.fields["t_get_opts_done_ns"].parse::<u64>()?;
            let finish = event.fields["t_finish_ns"].parse::<u64>()?;
            assert!(begin <= get_opts_done && get_opts_done <= finish);
        }
        Ok(())
    }

    #[tokio::test]
    async fn runbook_target_filter_preserves_parent_ids_and_emits_one_summary() -> anyhow::Result<()>
    {
        let (capture, _subscriber) = capture_runbook_traces();
        let (reader, _) = stream_reader(StreamBehavior::Normal).await?;
        let read_at_id = reader
            .diagnostic_instance_id()
            .ok_or_else(|| anyhow::anyhow!("diagnostic reader has no identity"))?;
        let parent = tracing::trace_span!(
            target: "vortex_file::read_ranges",
            "read_ranges",
            source_id = 53_u64,
            read_at_id,
            has_read_at_id = true,
            call_id = 11_u64,
        );
        let reads = {
            let _entered = parent.enter();
            reader.read_ranges(Arc::from([ReadAtRequest::new(7, 5, Alignment::new(1))]))
        };
        let results = reads.collect::<Vec<_>>().await;
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_ok());

        drop(reader);

        let events = capture.events.lock();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].parent_name, "read_ranges");
        assert_eq!(events[0].parent_fields["source_id"], "53");
        assert_eq!(
            events[0].parent_fields["read_at_id"],
            read_at_id.to_string()
        );
        assert_eq!(events[0].parent_fields["call_id"], "11");
        drop(events);

        let summaries = capture.summaries.lock();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].fields["scope"], "reader");
        assert_eq!(summaries[0].fields["read_at_id"], read_at_id.to_string());
        assert_eq!(summaries[0].fields["read_ranges_calls"], "1");
        assert_eq!(summaries[0].fields["children"], "1");
        assert_eq!(summaries[0].fields["terminals"], "1");
        Ok(())
    }

    #[tokio::test]
    async fn local_file_success_has_one_terminal() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("test.bin"), TEST_DATA)?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(directory.path())?);
        let reader = DiagnosticObjectStoreReadAt::new(
            store as Arc<dyn ObjectStore>,
            ObjectPath::from("test.bin"),
            TokioRuntime::current(),
        );
        let summary = Arc::clone(&reader.diagnostics);
        let buffer = reader.read_at(0, 6, Alignment::new(1)).await?;

        assert_eq!(buffer.to_host().await.as_slice(), b"object");
        assert_eq!(summary.file_payloads.load(Ordering::SeqCst), 1);
        assert_eq!(
            summary.blocking_queue_observations.load(Ordering::SeqCst),
            1
        );
        assert_eq!(summary.service_observations.load(Ordering::SeqCst), 1);
        assert_summary_consistent(&summary);
        let events = capture.events.lock();
        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(event.fields["outcome"], "success");
        assert_eq!(event.fields["payload"], "file");
        assert_eq!(event.fields["has_blocking_start"], "true");
        assert_ne!(event.fields["blocking_worker_id"], "0");
        assert_eq!(event.fields["has_blocking_worker_id"], "true");
        let begin = event.fields["t_begin_ns"].parse::<u64>()?;
        let get_opts_done = event.fields["t_get_opts_done_ns"].parse::<u64>()?;
        let blocking_start = event.fields["t_blocking_start_ns"].parse::<u64>()?;
        let finish = event.fields["t_finish_ns"].parse::<u64>()?;
        assert!(begin <= get_opts_done);
        assert!(get_opts_done <= blocking_start);
        assert!(blocking_start <= finish);
        Ok(())
    }

    #[tokio::test]
    async fn get_opts_error_has_one_terminal() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let reader = DiagnosticObjectStoreReadAt::new(
            Arc::new(InMemory::new()),
            ObjectPath::from("missing.bin"),
            TokioRuntime::current(),
        );
        let summary = Arc::clone(&reader.diagnostics);

        assert!(reader.read_at(0, 1, Alignment::new(1)).await.is_err());
        assert_eq!(summary.errors.load(Ordering::SeqCst), 1);
        assert_eq!(summary.no_payloads.load(Ordering::SeqCst), 1);
        assert_summary_consistent(&summary);
        assert_one_terminal(&capture, "error");
        Ok(())
    }

    #[tokio::test]
    async fn stream_error_and_short_result_each_have_one_terminal() -> anyhow::Result<()> {
        for behavior in [StreamBehavior::Error, StreamBehavior::Short] {
            let (capture, subscriber) = capture_traces();
            let (reader, _) = stream_reader(behavior).await?;
            let summary = Arc::clone(&reader.diagnostics);

            assert!(reader.read_at(0, 8, Alignment::new(1)).await.is_err());
            assert_eq!(summary.errors.load(Ordering::SeqCst), 1);
            assert_eq!(summary.stream_payloads.load(Ordering::SeqCst), 1);
            assert_summary_consistent(&summary);
            assert_one_terminal(&capture, "error");
            drop(subscriber);
        }
        Ok(())
    }

    #[tokio::test]
    async fn pending_stream_cancellation_has_one_terminal() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let (reader, pending_polls) = stream_reader(StreamBehavior::Pending).await?;
        let summary = Arc::clone(&reader.diagnostics);
        let read = reader.read_at(0, 8, Alignment::new(1));
        for _ in 0..100 {
            if pending_polls.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(pending_polls.load(Ordering::SeqCst), 1);
        drop(read);
        for _ in 0..100 {
            if summary.physical_cancellations.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }

        assert_eq!(summary.physical_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(summary.consumer_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(summary.stream_payloads.load(Ordering::SeqCst), 1);
        assert_summary_consistent(&summary);
        assert_one_terminal(&capture, "cancel");
        Ok(())
    }

    #[derive(Default)]
    struct PausedBlockingExecutor {
        blocking: parking_lot::Mutex<Vec<Box<dyn FnOnce() + Send + 'static>>>,
    }

    impl Executor for PausedBlockingExecutor {
        fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
            TokioAbortHandle::boxed(tokio::spawn(future).abort_handle())
        }

        fn spawn_io(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
            TokioAbortHandle::boxed(tokio::spawn(future).abort_handle())
        }

        fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            TokioAbortHandle::boxed(tokio::task::spawn_blocking(task).abort_handle())
        }

        fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
            self.blocking.lock().push(task);
            Box::new(NoopAbortHandle)
        }
    }

    struct TokioAbortHandle(tokio::task::AbortHandle);

    impl TokioAbortHandle {
        fn boxed(handle: tokio::task::AbortHandle) -> AbortHandleRef {
            Box::new(Self(handle))
        }
    }

    impl AbortHandle for TokioAbortHandle {
        fn abort(self: Box<Self>) {
            self.0.abort();
        }
    }

    struct NoopAbortHandle;

    impl AbortHandle for NoopAbortHandle {
        fn abort(self: Box<Self>) {}
    }

    #[tokio::test]
    async fn queued_blocking_cancellation_has_one_terminal() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let directory = tempfile::tempdir()?;
        std::fs::write(directory.path().join("test.bin"), TEST_DATA)?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(directory.path())?);
        let executor = Arc::new(PausedBlockingExecutor::default());
        let runtime = Arc::clone(&executor) as Arc<dyn Executor>;
        let reader = DiagnosticObjectStoreReadAt::new(
            store as Arc<dyn ObjectStore>,
            ObjectPath::from("test.bin"),
            Handle::new(Arc::downgrade(&runtime)),
        );
        let summary = Arc::clone(&reader.diagnostics);
        let read = reader.read_at(0, 6, Alignment::new(1));
        for _ in 0..100 {
            if !executor.blocking.lock().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(executor.blocking.lock().len(), 1);
        drop(read);
        for _ in 0..100 {
            if summary.consumer_cancellations.load(Ordering::SeqCst) != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }

        assert_eq!(summary.consumer_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(summary.physical_cancellations.load(Ordering::SeqCst), 0);
        let queued = std::mem::take(&mut *executor.blocking.lock());
        drop(queued);
        assert_eq!(summary.physical_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(summary.file_payloads.load(Ordering::SeqCst), 1);
        assert_summary_consistent(&summary);
        assert_one_terminal(&capture, "cancel");
        let events = capture.events.lock();
        assert_eq!(events[0].fields["consumer_cancelled"], "true");
        assert_eq!(events[0].fields["has_blocking_start"], "false");
        assert_eq!(events[0].fields["has_service"], "false");
        Ok(())
    }

    #[test]
    fn started_blocking_cancellation_keeps_physical_terminal_authoritative() -> anyhow::Result<()> {
        let (capture, _subscriber) = capture_traces();
        let summary = Arc::new(DiagnosticSummary::new(71, Arc::from("started.bin")));
        let parent = tracing::trace_span!("started_blocking_test");
        let mut terminal = ChildTerminal::new(
            Arc::clone(&summary),
            0,
            Arc::from("started.bin"),
            ReadAtRequest::new(0, 6, Alignment::new(1)),
            parent,
        );
        terminal.payload = Payload::File;
        terminal.t_blocking_enqueued_ns = Some(diagnostic_timestamp_ns());
        let consumer = terminal.consumer_guard();
        let (started_tx, started_rx) = std_mpsc::channel();
        let (finish_tx, finish_rx) = std_mpsc::channel();
        let worker = thread::spawn(move || {
            let service = Instant::now();
            terminal.mark_blocking_started(service);
            started_tx.send(())?;
            finish_rx.recv()?;
            terminal.service = Some(service.elapsed());
            terminal.success(6);
            Ok::<(), anyhow::Error>(())
        });

        started_rx.recv()?;
        drop(consumer);
        assert_eq!(summary.consumer_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(summary.terminals.load(Ordering::SeqCst), 0);
        finish_tx.send(())?;
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("blocking diagnostic worker panicked"))??;

        assert_eq!(summary.successes.load(Ordering::SeqCst), 1);
        assert_eq!(summary.physical_cancellations.load(Ordering::SeqCst), 0);
        assert_eq!(summary.terminals.load(Ordering::SeqCst), 1);
        assert_summary_consistent(&summary);
        let events = capture.events.lock();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].fields["outcome"], "success");
        assert_eq!(events[0].fields["consumer_cancelled"], "true");
        assert_eq!(events[0].fields["has_blocking_start"], "true");
        assert_eq!(events[0].fields["has_service"], "true");
        Ok(())
    }
}
