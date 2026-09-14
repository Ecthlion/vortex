// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
pub use vortex_io::ReadAtNowait;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides segment data to a [`crate::LayoutReader`].
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Stable identity for correlating diagnostics emitted by wrappers around this source.
    fn diagnostic_instance_id(&self) -> Option<u64> {
        None
    }

    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Register a segment for background reading and return its completion future.
    ///
    /// Sources with their own request queue may make this eligible before the future is polled.
    /// Other sources retain the default demand-driven behavior.
    fn request_background(&self, id: SegmentId) -> SegmentFuture {
        self.request(id)
    }

    /// Register a batch of segments for background reading.
    ///
    /// The returned futures correspond positionally to `ids`. Sources with a request queue may
    /// register the complete batch before making any member eligible, allowing adjacent requests
    /// to be coalesced even when the driver is running concurrently.
    fn request_background_batch(&self, ids: &[SegmentId]) -> Vec<SegmentFuture> {
        ids.iter().map(|&id| self.request_background(id)).collect()
    }

    /// Attempt to resolve a segment synchronously without waiting on storage.
    ///
    /// Sources that cannot guarantee non-blocking behavior return
    /// [`ReadAtNowait::Unsupported`].
    fn request_nowait(&self, _id: SegmentId) -> VortexResult<ReadAtNowait> {
        Ok(ReadAtNowait::Unsupported)
    }

    /// Whether planned reads should be submitted to a background driver immediately.
    ///
    /// Remote and file-backed sources use this to overlap IO and expose neighboring requests to
    /// coalescing before execution asks for their bytes. In-memory sources should retain the
    /// default so execution can resolve them inline without scheduler round trips.
    fn prefers_background_reads(&self) -> bool {
        false
    }
}

impl<S: SegmentSource + ?Sized> SegmentSource for std::sync::Arc<S> {
    fn diagnostic_instance_id(&self) -> Option<u64> {
        self.as_ref().diagnostic_instance_id()
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.as_ref().request(id)
    }

    fn request_background(&self, id: SegmentId) -> SegmentFuture {
        self.as_ref().request_background(id)
    }

    fn request_background_batch(&self, ids: &[SegmentId]) -> Vec<SegmentFuture> {
        self.as_ref().request_background_batch(ids)
    }

    fn request_nowait(&self, id: SegmentId) -> VortexResult<ReadAtNowait> {
        self.as_ref().request_nowait(id)
    }

    fn prefers_background_reads(&self) -> bool {
        self.as_ref().prefers_background_reads()
    }
}
