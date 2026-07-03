// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests of the WASM encoding pipeline using *real* compiled kernels.
//!
//! Each test pairs a host-side [`WasmEncoder`] with a kernel `.wasm` built from the matching
//! `vortex-wasm-guest` example (committed under `tests/fixtures/`, see that directory's `README`).
//! The kernels return their output as Arrow C Data Interface structs, which the host imports back
//! into a Vortex array — exercising the full [`WasmLayoutStrategy`] -> [`WasmReader`] -> Arrow
//! boundary, not just a hand-written WAT stand-in.

use std::sync::Arc;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
use vortex_layout::session::LayoutSession;
use vortex_wasm::IdentityEncoder;
use vortex_wasm::WasmEncoded;
use vortex_wasm::WasmEncoder;
use vortex_wasm::WasmLayoutStrategy;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedData;

/// The identity kernel: returns child 0 unchanged (`examples/identity-kernel`).
const IDENTITY_KERNEL: &[u8] = include_bytes!("fixtures/identity_kernel.wasm");
/// The Frame-of-Reference kernel for `i32` (`examples/for-kernel`).
const FOR_KERNEL: &[u8] = include_bytes!("fixtures/for_kernel.wasm");
/// The `vortex.fastlanes.bitpacked` kernel for `i32` (`examples/bitpacked-kernel`).
const BITPACKED_KERNEL: &[u8] = include_bytes!("fixtures/bitpacked_kernel.wasm");
/// The FSST string decoder kernel (`examples/fsst-kernel`).
const FSST_KERNEL: &[u8] = include_bytes!("fixtures/fsst_kernel.wasm");

/// Write `array` through a [`WasmLayoutStrategy`] with `kernel`/`encoder`, then decode the whole
/// column back through a [`WasmReader`].
fn round_trip(
    kernel: &'static [u8],
    encoding_id: &str,
    encoder: Arc<dyn WasmEncoder>,
    array: ArrayRef,
) -> ArrayRef {
    block_on(|handle| async move {
        let session = array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle);

        let dtype = array.dtype().clone();
        let strategy = WasmLayoutStrategy::new(
            ByteBuffer::from(kernel.to_vec()),
            encoding_id,
            encoder,
            Arc::new(FlatLayoutStrategy::default()) as Arc<dyn LayoutStrategy>,
        );

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let layout = strategy
            .write_stream(
                ArrayContext::empty(),
                Arc::<TestSegments>::clone(&segments),
                SequentialStreamAdapter::new(dtype, array.to_array_stream().sequenced(ptr))
                    .sendable(),
                eof,
                &session,
            )
            .await
            .expect("write");

        let row_count = layout.row_count();
        let reader = layout
            .new_reader(encoding_id.into(), segments, &session, &Default::default())
            .expect("reader");
        reader
            .projection_evaluation(
                &(0..row_count),
                &root(),
                MaskFuture::new_true(row_count as usize),
            )
            .expect("projection")
            .await
            .expect("decode")
    })
}

/// Collect the validity of `array` as a `Vec<bool>` of length `len`.
fn validity_bools(array: &ArrayRef, len: usize) -> Vec<bool> {
    let mut ctx = array_session().create_execution_ctx();
    let bits = array
        .validity()
        .expect("validity")
        .execute_mask(len, &mut ctx)
        .expect("mask")
        .to_bit_buffer();
    (0..len).map(|i| bits.value(i)).collect()
}

#[test]
fn identity_round_trips_non_nullable() {
    let values = vec![10i32, 20, 30, 40];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let out = round_trip(
        IDENTITY_KERNEL,
        "test.identity",
        Arc::new(IdentityEncoder),
        array,
    );

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

#[test]
fn identity_round_trips_nullable() {
    // Positions 1 and 3 are null. The identity kernel preserves the child's validity bitmap, so the
    // whole nullable column survives the Arrow boundary in both directions.
    let validity = Validity::from_iter([true, false, true, false, true]);
    let values = [100i32, 200, 300, 400, 500];
    let array = PrimitiveArray::new(Buffer::copy_from(values), validity).into_array();
    let out = round_trip(
        IDENTITY_KERNEL,
        "test.identity",
        Arc::new(IdentityEncoder),
        array,
    );

    assert_eq!(out.len(), 5);
    assert_eq!(
        validity_bools(&out, 5),
        vec![true, false, true, false, true]
    );
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

/// Frame-of-Reference encoder for `i32`: store the minimum as the payload reference and the
/// per-element deltas as the child.
struct ForEncoder;

impl WasmEncoder for ForEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!("ForEncoder only supports i32, got {}", primitive.ptype());
        }
        let values = primitive.as_slice::<i32>();
        let reference = values.iter().copied().min().unwrap_or(0);
        let deltas: Vec<i32> = values.iter().map(|v| v - reference).collect();

        let payload = ByteBuffer::from(reference.to_le_bytes().to_vec());
        let child =
            PrimitiveArray::new(Buffer::copy_from(&deltas), Validity::NonNullable).into_array();
        Ok(WasmEncoded {
            payload,
            child: Some(child),
        })
    }
}

#[test]
fn for_round_trips() {
    let values = vec![1000i32, 1005, 1002, 1010, 1001, 1000, 1234];
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let out = round_trip(FOR_KERNEL, "test.for", Arc::new(ForEncoder), array);

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

/// Encoder for the real `vortex.fastlanes.bitpacked` encoding: packs with the native
/// [`BitPackedData::encode`] (the same code the `BitPacked` VTable runs), then serializes the
/// resulting parts — bit width, offset, patches, and the FastLanes-packed buffer — into the
/// payload the kernel consumes. The kernel decodes those bytes with the same `fastlanes` crate,
/// so the two sides share semantics by construction.
struct BitPackedEncoder {
    bit_width: u8,
}

/// Serialize an encoded [`BitPackedArray`]'s parts into the kernel payload.
fn bitpacked_payload(bp: &BitPackedArray, ctx: &mut ExecutionCtx) -> VortexResult<ByteBuffer> {
    let packed = bp.packed().clone().try_to_host_sync()?;
    let (positions, patch_values): (Vec<u32>, Vec<i32>) = match bp.patches() {
        Some(patches) => {
            let indices = patches
                .indices()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_primitive();
            let values = patches
                .values()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_primitive();
            let positions = match indices.ptype() {
                PType::U8 => indices
                    .as_slice::<u8>()
                    .iter()
                    .map(|&i| i as usize)
                    .collect::<Vec<_>>(),
                PType::U16 => indices
                    .as_slice::<u16>()
                    .iter()
                    .map(|&i| i as usize)
                    .collect(),
                PType::U32 => indices
                    .as_slice::<u32>()
                    .iter()
                    .map(|&i| i as usize)
                    .collect(),
                PType::U64 => indices
                    .as_slice::<u64>()
                    .iter()
                    .map(|&i| usize::try_from(i).expect("patch index fits usize"))
                    .collect(),
                other => vortex_bail!("unexpected patch index ptype {other}"),
            };
            let positions = positions
                .into_iter()
                .map(|i| u32::try_from(i - patches.offset()).expect("patch position fits u32"))
                .collect();
            (positions, values.as_slice::<i32>().to_vec())
        }
        None => (Vec::new(), Vec::new()),
    };

    let mut payload = Vec::with_capacity(12 + positions.len() * 8 + packed.len());
    payload.push(bp.bit_width());
    payload.push(0);
    payload.extend_from_slice(&bp.offset().to_le_bytes());
    payload.extend_from_slice(&(bp.as_ref().len() as u32).to_le_bytes());
    payload.extend_from_slice(&(positions.len() as u32).to_le_bytes());
    for position in &positions {
        payload.extend_from_slice(&position.to_le_bytes());
    }
    for value in &patch_values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    payload.extend_from_slice(packed.as_slice());
    Ok(ByteBuffer::from(payload))
}

impl WasmEncoder for BitPackedEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let primitive = chunk.execute::<Canonical>(ctx)?.into_primitive();
        if primitive.ptype() != PType::I32 {
            vortex_bail!("BitPackedEncoder example only supports i32");
        }
        let bp = BitPackedData::encode(&primitive.into_array(), self.bit_width, ctx)?;
        Ok(WasmEncoded {
            payload: bitpacked_payload(&bp, ctx)?,
            child: None,
        })
    }
}

#[test]
fn bitpacked_round_trips() {
    // 3000 values within 6 bits: two full FastLanes chunks plus a partial trailer.
    let values: Vec<i32> = (0..3000).map(|i| i % 64).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let out = round_trip(
        BITPACKED_KERNEL,
        "vortex.fastlanes.bitpacked",
        Arc::new(BitPackedEncoder { bit_width: 6 }),
        array,
    );

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

#[test]
fn bitpacked_with_patches_round_trips() {
    // 1% of values exceed the 6-bit budget, so the native encoder emits patches.
    let values: Vec<i32> = (0..3000)
        .map(|i| if i % 100 == 0 { 1_000_000 + i } else { i % 64 })
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();

    // Prove this data actually exercises the patch path in the native encoding.
    let mut ctx = array_session().create_execution_ctx();
    let bp = BitPackedData::encode(&array, 6, &mut ctx).expect("encode");
    assert!(bp.patches().is_some(), "expected patches for the outliers");

    let out = round_trip(
        BITPACKED_KERNEL,
        "vortex.fastlanes.bitpacked",
        Arc::new(BitPackedEncoder { bit_width: 6 }),
        array,
    );

    assert_eq!(out.len(), values.len());
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(out.buffers()[0].as_ref(), expected.as_slice());
}

/// FSST encoder for utf8 strings: train a symbol table with the `fsst` crate, compress every
/// string, and pack the whole encoded form into the payload (see the kernel's doc for the layout).
/// The compressor stays host-side; only the tiny table-walk decoder ships in the file.
struct FsstEncoder;

impl FsstEncoder {
    fn encode_strings(strings: &[&[u8]]) -> ByteBuffer {
        let compressor = fsst::Compressor::train(&strings.to_vec());
        let symbols = compressor.symbol_table();
        let lengths = compressor.symbol_lengths();

        let mut codes = Vec::new();
        let mut code_offsets: Vec<u32> = Vec::with_capacity(strings.len() + 1);
        code_offsets.push(0);
        for string in strings {
            codes.extend_from_slice(&compressor.compress(string));
            code_offsets.push(u32::try_from(codes.len()).expect("codes fit in u32"));
        }

        let mut payload = Vec::new();
        payload.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
        for symbol in symbols {
            payload.extend_from_slice(&symbol.to_u64().to_le_bytes());
        }
        payload.extend_from_slice(lengths);
        payload.extend_from_slice(&(strings.len() as u32).to_le_bytes());
        for offset in &code_offsets {
            payload.extend_from_slice(&offset.to_le_bytes());
        }
        payload.extend_from_slice(&codes);
        ByteBuffer::from(payload)
    }
}

impl WasmEncoder for FsstEncoder {
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        let strings = chunk.execute::<Canonical>(ctx)?.into_varbinview();
        let buffers: Vec<_> = (0..strings.len()).map(|i| strings.bytes_at(i)).collect();
        let lines: Vec<&[u8]> = buffers.iter().map(|b| b.as_slice()).collect();
        Ok(WasmEncoded {
            payload: Self::encode_strings(&lines),
            child: None,
        })
    }
}

#[test]
fn fsst_round_trips() {
    let strings: Vec<String> = (0..512)
        .map(|i| format!("https://vortex.dev/docs/page-{}?ref=benchmark", i % 100))
        .collect();
    let array = VarBinViewArray::from_iter_str(strings.iter()).into_array();
    let out = round_trip(FSST_KERNEL, "test.fsst", Arc::new(FsstEncoder), array);

    assert_eq!(out.len(), strings.len());
    let mut ctx = array_session().create_execution_ctx();
    let decoded = out.execute::<Canonical>(&mut ctx).expect("canonical");
    let decoded = decoded.into_varbinview();
    for (i, expected) in strings.iter().enumerate() {
        assert_eq!(decoded.bytes_at(i).as_slice(), expected.as_bytes());
    }
}

#[test]
fn fsst_reduces_size() {
    let strings: Vec<String> = (0..512)
        .map(|i| format!("https://vortex.dev/docs/page-{}?ref=benchmark", i % 100))
        .collect();
    let lines: Vec<&[u8]> = strings.iter().map(|s| s.as_bytes()).collect();
    let raw: usize = lines.iter().map(|l| l.len()).sum();
    let payload = FsstEncoder::encode_strings(&lines);
    assert!(
        payload.len() * 2 < raw,
        "expected >2x reduction: payload={} raw={raw}",
        payload.len()
    );
}

