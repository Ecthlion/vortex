// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A minimal protobuf wire-format reader, so kernels can parse the **real** prost-encoded
//! encoding metadata (e.g. `BitPackedMetadata`, `FSSTMetadata`) without pulling `prost` (or any
//! other dependency) into the guest.
//!
//! Only what Vortex metadata needs: varint (wire type 0), length-delimited (wire type 2), and
//! skipping over fixed 32/64-bit fields. Nested messages arrive as [`Field::Bytes`] and are parsed
//! with a nested [`ProtoReader`].

use crate::error::GuestError;
use crate::error::GuestResult;

/// A decoded protobuf field value.
pub enum Field<'a> {
    /// Wire type 0.
    Varint(u64),
    /// Wire type 2 (nested messages, strings, packed fields).
    Bytes(&'a [u8]),
    /// Wire type 1 (fixed 64-bit), raw little-endian bits.
    Fixed64(u64),
    /// Wire type 5 (fixed 32-bit), raw little-endian bits.
    Fixed32(u32),
}

/// Iterates the fields of one protobuf message.
pub struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    /// Read fields from a serialized message.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn varint(&mut self) -> GuestResult<u64> {
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let byte = *self
                .buf
                .get(self.pos)
                .ok_or(GuestError::new("proto: truncated varint"))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                return Err(GuestError::new("proto: varint too long"));
            }
        }
    }

    fn take(&mut self, n: usize) -> GuestResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.buf.len())
            .ok_or(GuestError::new("proto: truncated field"))?;
        let bytes = &self.buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// Return the next `(field_number, value)`, or `None` at end of message.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> GuestResult<Option<(u32, Field<'a>)>> {
        if self.pos >= self.buf.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = u32::try_from(key >> 3).map_err(|_| GuestError::new("proto: bad field"))?;
        let value = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 => {
                let bytes = self.take(8)?;
                Field::Fixed64(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
            }
            2 => {
                let len = usize::try_from(self.varint()?)
                    .map_err(|_| GuestError::new("proto: bad length"))?;
                Field::Bytes(self.take(len)?)
            }
            5 => {
                let bytes = self.take(4)?;
                Field::Fixed32(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
            }
            _ => return Err(GuestError::new("proto: unsupported wire type")),
        };
        Ok(Some((field, value)))
    }
}
