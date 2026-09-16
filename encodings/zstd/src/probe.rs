// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Zstd probes retain every decompressed frame and index the compacted, non-null value positions.

use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::RepeatedArrayProbe;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::dtype::DType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Zstd;
use crate::ZstdBuffers;
use crate::ZstdData;
use crate::array::ViewLen;
use crate::array::unsliced_validity;
use crate::array::zstd_value_len;

/// Rows per entry of the non-null rank index.
const RANK_STRIDE: usize = 512;

/// State retained by repeated [`Zstd`] probes. Construction performs no allocation.
///
/// The mask and rank index use unsliced logical rows; frame boundaries use compacted value
/// positions, since nulls are not stored in the frames. Every frame decompressed so far is
/// retained, so the state grows with the frames touched and never beyond the decompressed array.
#[derive(Default)]
pub struct ZstdProbeState {
    validity: Option<Mask>,
    rank: Vec<usize>,
    frames: Vec<Frame>,
    last_frame: usize,
}

struct Frame {
    values: Range<usize>,
    decoded: Option<DecodedFrame>,
}

/// A decompressed frame, laid out so that a row lands on a value without another walk.
enum DecodedFrame {
    /// Fixed-width values, indexed directly.
    Primitive(PrimitiveArray),
    /// Length-prefixed values, alongside the byte offset of each value's prefix.
    VarBin {
        bytes: ByteBuffer,
        offsets: Vec<usize>,
    },
}

pub(crate) fn scalar_at(
    array: ArrayView<'_, Zstd>,
    index: usize,
    state: Option<&mut ZstdProbeState>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let Some(state) = state else {
        // A one-off read decompresses only the frames holding the single row, as before.
        let validity = unsliced_validity(array);
        return array
            .data()
            .with_slice(index, index + 1)
            .decompress(array.dtype(), &validity, ctx)?
            .execute_scalar(0, ctx);
    };

    let logical_index = array.data().slice_start() + index;
    let value_index = value_index(array, state, logical_index, ctx)?;
    let byte_width = ZstdData::byte_width(array.dtype());

    if state.frames.is_empty() {
        let n_valid = state
            .validity
            .as_ref()
            .map(Mask::true_count)
            .ok_or_else(|| vortex_err!("Zstd probe validity was just resolved"))?;
        state.frames = frame_ranges(array, byte_width, n_valid)?;
    }
    let frame_index = match state.frames.get(state.last_frame) {
        Some(frame) if frame.values.contains(&value_index) => state.last_frame,
        _ => state
            .frames
            .partition_point(|frame| frame.values.end <= value_index),
    };
    state.last_frame = frame_index;

    let frame = state
        .frames
        .get_mut(frame_index)
        .ok_or_else(|| vortex_err!("Missing zstd frame for value {value_index}"))?;
    let offset = value_index.checked_sub(frame.values.start).ok_or_else(|| {
        vortex_err!("Corrupt zstd metadata: frame value ranges are not ascending")
    })?;
    let n_values = frame.values.len();
    let decoded = match &mut frame.decoded {
        Some(decoded) => decoded,
        slot @ None => slot.insert(decode_frame(array, frame_index, byte_width, n_values, ctx)?),
    };
    decoded.scalar(offset, array.dtype())
}

/// The position of `logical_index` among the non-null rows, which is where its value is stored.
///
/// Nulls are not stored in the frames, so the rank index is what turns a row into a value. It is
/// built once per probe, then each read counts within one [`RANK_STRIDE`] row block.
fn value_index(
    array: ArrayView<'_, Zstd>,
    state: &mut ZstdProbeState,
    logical_index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    let mask = match &mut state.validity {
        Some(mask) => mask,
        slot @ None => {
            slot.insert(unsliced_validity(array).execute_mask(array.data().unsliced_n_rows(), ctx)?)
        }
    };
    match mask {
        Mask::AllTrue(_) => Ok(logical_index),
        // Probe dispatch resolves validity before reaching an encoding.
        Mask::AllFalse(_) => vortex_bail!("Zstd probe read null row {logical_index}"),
        Mask::Values(values) => {
            let bits = values.bit_buffer();
            vortex_ensure!(
                logical_index < bits.len(),
                "Zstd row {logical_index} is out of bounds of the {} row validity",
                bits.len()
            );
            if state.rank.is_empty() {
                state.rank.reserve(bits.len().div_ceil(RANK_STRIDE));
                let mut count = 0;
                for start in (0..bits.len()).step_by(RANK_STRIDE) {
                    state.rank.push(count);
                    count += bits.count_range(start, (start + RANK_STRIDE).min(bits.len()));
                }
            }
            let block = logical_index / RANK_STRIDE;
            let base = state
                .rank
                .get(block)
                .ok_or_else(|| vortex_err!("Zstd rank index is missing block {block}"))?;
            Ok(base + bits.count_range(block * RANK_STRIDE, logical_index))
        }
    }
}

/// The compacted value range each frame holds, in the order the frames are stored.
///
/// `n_valid` covers the metadata that predates per-frame value counts, matching what the
/// decompression path accepts.
fn frame_ranges(
    array: ArrayView<'_, Zstd>,
    byte_width: usize,
    n_valid: usize,
) -> VortexResult<Vec<Frame>> {
    let data = array.data();
    let mut frames = Vec::with_capacity(data.frames.len());
    let mut start = 0usize;
    for frame_meta in data.metadata.frames.iter().take(data.frames.len()) {
        let n_values = if frame_meta.n_values != 0 {
            usize::try_from(frame_meta.n_values).map_err(|_| {
                vortex_err!(
                    "Zstd frame value count {} does not fit in a usize",
                    frame_meta.n_values
                )
            })?
        } else if array.dtype().is_primitive() {
            // Possibly older primitive-only metadata that just didn't store this. Fixed-width
            // values make the byte count an exact value count.
            let uncompressed_size =
                usize::try_from(frame_meta.uncompressed_size).map_err(|_| {
                    vortex_err!(
                        "Zstd frame uncompressed size {} does not fit in a usize",
                        frame_meta.uncompressed_size
                    )
                })?;
            uncompressed_size / byte_width
        } else {
            // The same fallback would read a byte count as a value count for variable-width
            // values, which misattributes values to frames. A single frame holds every stored
            // value, so that case is still recoverable; anything else is not.
            vortex_ensure!(
                data.frames.len() == 1,
                "Zstd frame metadata for a variable-width array is missing its value count"
            );
            n_valid
        };
        let end = start.checked_add(n_values).ok_or_else(|| {
            vortex_err!("Corrupt zstd metadata: frame value counts overflow a usize")
        })?;
        frames.push(Frame {
            values: start..end,
            decoded: None,
        });
        start = end;
    }
    Ok(frames)
}

fn decode_frame(
    array: ArrayView<'_, Zstd>,
    frame_index: usize,
    byte_width: usize,
    n_values: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecodedFrame> {
    let bytes = array.data().decompress_frame(frame_index, byte_width)?;
    match array.dtype() {
        DType::Primitive(ptype, _) => {
            vortex_ensure!(
                bytes.len().is_multiple_of(byte_width),
                "Corrupt zstd metadata: frame {frame_index} holds {} bytes, which is not a whole \
                 number of {byte_width} byte values",
                bytes.len()
            );
            let n_rows = bytes.len() / byte_width;
            Ok(DecodedFrame::Primitive(
                PrimitiveArray::from_values_byte_buffer(
                    bytes,
                    *ptype,
                    Validity::NonNullable,
                    n_rows,
                    ctx,
                ),
            ))
        }
        DType::Binary(_) | DType::Utf8(_) => {
            // Walking the length prefixes is a dependent load chain, so it runs once for the whole
            // frame and every later read of it indexes straight to a value. The walk is what
            // bounds the offsets, and each value needs a prefix, so the capacity the untrusted
            // value count asks for is held to what the frame could hold.
            let mut offsets =
                Vec::with_capacity(n_values.min(bytes.len() / size_of::<ViewLen>() + 1));
            let mut offset = 0;
            for _ in 0..n_values {
                offsets.push(offset);
                offset += size_of::<ViewLen>() + zstd_value_len(bytes.as_slice(), offset)?;
                vortex_ensure!(
                    offset <= bytes.len(),
                    "Corrupt zstd values: walking frame {frame_index} ended at offset {offset}, \
                     past the end of the {} byte frame buffer",
                    bytes.len()
                );
            }
            Ok(DecodedFrame::VarBin { bytes, offsets })
        }
        dtype => vortex_bail!("Unsupported dtype for Zstd array: {dtype}"),
    }
}

impl DecodedFrame {
    /// The `offset`th value of this frame, as a scalar of `dtype`.
    fn scalar(&self, offset: usize, dtype: &DType) -> VortexResult<Scalar> {
        match self {
            Self::Primitive(values) => {
                vortex_ensure!(
                    offset < values.len(),
                    "Corrupt zstd metadata: value {offset} is past the {} values the frame holds",
                    values.len()
                );
                Ok(match_each_native_ptype!(values.ptype(), |T| {
                    Scalar::primitive(values.as_slice::<T>()[offset], dtype.nullability())
                }))
            }
            Self::VarBin { bytes, offsets } => {
                let prefix = *offsets.get(offset).ok_or_else(|| {
                    vortex_err!(
                        "Corrupt zstd metadata: value {offset} is past the {} values the frame \
                         holds",
                        offsets.len()
                    )
                })?;
                let start = prefix + size_of::<ViewLen>();
                let end = start
                    .checked_add(zstd_value_len(bytes.as_slice(), prefix)?)
                    .filter(|end| *end <= bytes.len())
                    .ok_or_else(|| {
                        vortex_err!(
                            "Corrupt zstd values: value {offset} runs past the end of the {} byte \
                             frame buffer",
                            bytes.len()
                        )
                    })?;
                Ok(varbin_scalar(bytes.slice(start..end), dtype))
            }
        }
    }
}

/// State retained by repeated [`ZstdBuffers`] probes.
///
/// A read has to decompress every buffer of the wrapped array, so the probe keeps the inner
/// array and reads it through a probe of its own; a one-off read still pays for the whole
/// decompression.
#[derive(Default)]
pub struct ZstdBuffersProbeState {
    inner: Option<RepeatedArrayProbe>,
}

pub(crate) fn buffers_scalar_at(
    state: &mut ProbeState<'_, ZstdBuffers>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let array = state.array().into_owned();
    let Some(retained) = state.retained() else {
        return ZstdBuffers::decompress_and_build_inner(&array, ctx.session())?
            .execute_scalar(index, ctx);
    };
    let inner = match &mut retained.inner {
        Some(inner) => inner,
        slot @ None => slot.insert(RepeatedArrayProbe::new(
            ZstdBuffers::decompress_and_build_inner(&array, ctx.session())?,
        )),
    };
    inner.execute_scalar(index, ctx)
}

#[cfg(test)]
mod tests;
