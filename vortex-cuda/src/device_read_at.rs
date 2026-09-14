// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::error::VortexResult;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;

use crate::stream::VortexCudaStream;

/// A wrapper that uses an allocator to produce the returned buffer handle.
#[derive(Clone)]
pub struct CopyDeviceReadAt<T: VortexReadAt + Clone> {
    read: T,
    stream: VortexCudaStream,
}

impl<T: VortexReadAt + Clone> CopyDeviceReadAt<T> {
    pub fn new(read: T, stream: VortexCudaStream) -> Self {
        Self { read, stream }
    }
}

impl<T: VortexReadAt + Clone> VortexReadAt for CopyDeviceReadAt<T> {
    fn diagnostic_instance_id(&self) -> Option<u64> {
        forwarded_diagnostic_instance_id(&self.read)
    }

    fn uri(&self) -> Option<&Arc<str>> {
        self.read.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.read.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.read.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.read.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let read = self.read.clone();
        let stream = self.stream.clone();
        async move {
            let handle = read.read_at(offset, length, alignment).await?;
            if handle.is_on_device() {
                return Ok(handle);
            }

            let host_buffer = handle.as_host().clone();

            stream.copy_to_device(host_buffer)?.await
        }
        .boxed()
    }
}

fn forwarded_diagnostic_instance_id(read: &impl VortexReadAt) -> Option<u64> {
    read.diagnostic_instance_id()
}

#[cfg(test)]
mod tests {
    use vortex::array::buffer::BufferHandle;
    use vortex::buffer::ByteBuffer;

    use super::*;

    #[derive(Clone)]
    struct IdentifiedReadAt;

    impl VortexReadAt for IdentifiedReadAt {
        fn diagnostic_instance_id(&self) -> Option<u64> {
            Some(73)
        }

        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(0) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { Ok(BufferHandle::new_host(ByteBuffer::from(Vec::<u8>::new()))) }.boxed()
        }
    }

    #[test]
    fn copy_device_reader_forwards_diagnostic_instance_id() {
        assert_eq!(
            forwarded_diagnostic_instance_id(&IdentifiedReadAt),
            Some(73)
        );
    }
}
