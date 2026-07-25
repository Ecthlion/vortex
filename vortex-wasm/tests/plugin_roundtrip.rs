// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests of wasm-backed encodings against the **real serialized format**.
//!
//! Each test serializes an array with the *native* encoding (the exact bytes a Vortex file would
//! contain), then deserializes it in a session that lacks the native encoding but has the
//! embedded kernel registered via [`register_wasm_encodings`]. The kernel — compiled from the
//! crate living alongside the native encoding (see `tests/fixtures/`) — must reproduce the
//! original values from those same bytes: parity with the native decoder by construction.

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedData;
use vortex_fsst::FSST;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_runend::RunEnd;
use vortex_runend::RunEndArrayExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;
use vortex_wasm::register_wasm_encodings;

/// The `fastlanes.bitpacked` kernel (`encodings/fastlanes/wasm`).
const BITPACKED_KERNEL: &[u8] = include_bytes!("fixtures/bitpacked_kernel.wasm");
/// The `vortex.fsst` kernel (`encodings/fsst/wasm`).
const FSST_KERNEL: &[u8] = include_bytes!("fixtures/fsst_kernel.wasm");
/// The `vortex.runend` kernel (`encodings/runend/wasm`).
const RUNEND_KERNEL: &[u8] = include_bytes!("fixtures/runend_kernel.wasm");

/// Serialize `array` exactly as a file would, then deserialize it in a session that has no native
/// decoder for it — only the given wasm kernel.
fn round_trip_via_wasm(
    array: ArrayRef,
    write_session: &VortexSession,
    kernel_id: &str,
    kernel: &[u8],
) -> VortexResult<ArrayRef> {
    let dtype = array.dtype().clone();
    let len = array.len();

    let array_ctx = ArrayContext::empty();
    let serialized = array.serialize(&array_ctx, write_session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }

    // A fresh session has the canonical encodings but not the external native encoding; merging
    // the kernel makes the unknown encoding decodable.
    let read_session = array_session();
    let registered = register_wasm_encodings(
        &read_session,
        [(kernel_id.to_string(), ByteBuffer::from(kernel.to_vec()))],
    )?;
    assert_eq!(registered, vec![kernel_id.to_string()]);

    SerializedArray::try_from(concat.freeze())?.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &read_session,
    )
}

/// A session with the native fastlanes/fsst encodings registered (the "writer").
fn native_session() -> VortexSession {
    let session = array_session();
    session.get::<ArraySession>().register(BitPacked);
    session.get::<ArraySession>().register(FSST);
    session.get::<ArraySession>().register(RunEnd);
    session
}

#[test]
fn bitpacked_decodes_via_wasm() -> VortexResult<()> {
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    // 3000 values within 6 bits: two full FastLanes chunks plus a partial trailer, no patches.
    let values: Vec<i32> = (0..3000).map(|i| i % 64).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let packed = BitPackedData::encode(&array, 6, &mut ctx)?;
    assert!(packed.patches().is_none());

    let decoded = round_trip_via_wasm(
        packed.into_array(),
        &session,
        "fastlanes.bitpacked",
        BITPACKED_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn bitpacked_with_patches_decodes_via_wasm() -> VortexResult<()> {
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    // 1% of values exceed the 6-bit budget, so the native encoder emits patches.
    let values: Vec<i32> = (0..3000)
        .map(|i| if i % 100 == 0 { 1_000_000 + i } else { i % 64 })
        .collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let packed = BitPackedData::encode(&array, 6, &mut ctx)?;
    assert!(packed.patches().is_some(), "expected patches for outliers");

    let decoded = round_trip_via_wasm(
        packed.into_array(),
        &session,
        "fastlanes.bitpacked",
        BITPACKED_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn bitpacked_nullable_decodes_via_wasm() -> VortexResult<()> {
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let values: Vec<i32> = (0..2000).map(|i| i % 32).collect();
    let validity = Validity::from_iter((0..2000).map(|i| i % 7 != 0));
    let array = PrimitiveArray::new(Buffer::copy_from(&values), validity).into_array();
    let packed = BitPackedData::encode(&array, 5, &mut ctx)?;

    let decoded = round_trip_via_wasm(
        packed.into_array(),
        &session,
        "fastlanes.bitpacked",
        BITPACKED_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn fsst_decodes_via_wasm() -> VortexResult<()> {
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let strings: Vec<String> = (0..512)
        .map(|i| format!("https://vortex.dev/docs/page-{}?ref=benchmark", i % 100))
        .collect();
    let array = VarBinViewArray::from_iter_str(strings.iter()).into_array();
    let compressor = fsst_train_compressor(&array, &mut ctx)?;
    let compressed = fsst_compress(&array, &compressor, &mut ctx)?;

    let decoded = round_trip_via_wasm(
        compressed.into_array(),
        &session,
        "vortex.fsst",
        FSST_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn fsst_nullable_decodes_via_wasm() -> VortexResult<()> {
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let strings: Vec<Option<String>> = (0..256)
        .map(|i| (i % 5 != 0).then(|| format!("value-{}-{}", i % 40, i % 7)))
        .collect();
    let array =
        VarBinViewArray::from_iter_nullable_str(strings.iter().map(|s| s.as_deref())).into_array();
    let compressor = fsst_train_compressor(&array, &mut ctx)?;
    let compressed = fsst_compress(&array, &compressor, &mut ctx)?;

    let decoded = round_trip_via_wasm(
        compressed.into_array(),
        &session,
        "vortex.fsst",
        FSST_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Run-end: the structural case. The kernel never sees the values child — it only expands the run
// ends into gather indices and names the child, so the host gathers it lazily. That is what lets
// these tests cover dtypes the kernel contains no code for.
// ---------------------------------------------------------------------------------------------

#[test]
fn runend_primitive_decodes_via_wasm() -> VortexResult<()> {
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let values: Vec<i32> = (0..4000).map(|i| (i / 37) % 11).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let encoded = RunEnd::encode(array.clone(), &mut ctx)?;

    let decoded = round_trip_via_wasm(
        encoded.into_array(),
        &session,
        "vortex.runend",
        RUNEND_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn runend_strings_decode_via_wasm() -> VortexResult<()> {
    // The payoff of ChildMode::Reference: the kernel has no string code at all, and strings never
    // enter guest memory. The native decoder needs a dedicated varbinview implementation for this.
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let strings: Vec<String> = (0..1200)
        .map(|i| format!("region-{}", (i / 53) % 9))
        .collect();
    let array = VarBinViewArray::from_iter_str(strings.iter()).into_array();
    // Run-end encode by hand: `RunEndArray::encode` only accepts primitives.
    let mut ends: Vec<u64> = Vec::new();
    let mut run_values: Vec<&String> = Vec::new();
    for (i, s) in strings.iter().enumerate() {
        if run_values.last().is_none_or(|prev| *prev != s) {
            if i > 0 {
                ends.push(i as u64);
            }
            run_values.push(s);
        }
    }
    ends.push(strings.len() as u64);
    let encoded = RunEnd::try_new(
        PrimitiveArray::new(Buffer::copy_from(&ends), Validity::NonNullable).into_array(),
        VarBinViewArray::from_iter_str(run_values.iter().map(|s| s.as_str())).into_array(),
        &mut ctx,
    )?;

    let decoded = round_trip_via_wasm(
        encoded.into_array(),
        &session,
        "vortex.runend",
        RUNEND_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn runend_sliced_decodes_via_wasm() -> VortexResult<()> {
    // A sliced run-end array carries a non-zero `offset`, which the kernel must apply when
    // expanding run ends (mirroring `trimmed_ends_iter`).
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    // Runs of 100, so an offset of 50 falls inside the first run (which
    // `try_new_offset_length` requires: the first run end must not precede the offset).
    let values: Vec<i32> = (0..3000).map(|i| (i / 100) % 7).collect();
    let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array();
    let encoded = RunEnd::encode(array.clone(), &mut ctx)?;

    // Build the sliced view directly so the `offset` field is definitely exercised.
    let (start, stop) = (50usize, 2593usize);
    let sliced = RunEnd::try_new_offset_length(
        encoded.ends().clone(),
        encoded.values().clone(),
        start,
        stop - start,
        &mut ctx,
    )?;
    assert!(sliced.offset() > 0, "expected a non-zero run-end offset");
    let expected = array.slice(start..stop)?;

    let decoded = round_trip_via_wasm(
        sliced.into_array(),
        &session,
        "vortex.runend",
        RUNEND_KERNEL,
    )?;
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn runend_nullable_decodes_via_wasm() -> VortexResult<()> {
    // Run-end's output validity is the values' validity gathered through the same runs; the host
    // `take` reproduces that without the kernel touching validity at all.
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let values: Vec<i32> = (0..2400).map(|i| (i / 31) % 9).collect();
    let validity = Validity::from_iter((0..2400).map(|i| (i / 31) % 4 != 0));
    let array = PrimitiveArray::new(Buffer::copy_from(&values), validity).into_array();
    let encoded = RunEnd::encode(array.clone(), &mut ctx)?;

    let decoded = round_trip_via_wasm(
        encoded.into_array(),
        &session,
        "vortex.runend",
        RUNEND_KERNEL,
    )?;
    assert_arrays_eq!(decoded, array, &mut ctx);
    Ok(())
}

#[test]
fn lying_kernel_errors_rather_than_aborting() -> VortexResult<()> {
    // A kernel's child declarations and gather indices are untrusted file data. Point the run-end
    // kernel at FSST-encoded bytes: its declared children will not match what is actually there.
    // The requirement is that this surfaces as a VortexError — before this work the mismatch hit
    // an `assert_eq!` inside `SerializedArray::decode` and aborted the process.
    let session = native_session();
    let mut ctx = session.create_execution_ctx();

    let strings: Vec<String> = (0..64).map(|i| format!("value-{}", i % 8)).collect();
    let array = VarBinViewArray::from_iter_str(strings.iter()).into_array();
    let compressor = fsst_train_compressor(&array, &mut ctx)?;
    let compressed = fsst_compress(&array, &compressor, &mut ctx)?.into_array();

    let dtype = compressed.dtype().clone();
    let len = compressed.len();
    let array_ctx = ArrayContext::empty();
    let serialized = compressed.serialize(&array_ctx, &session, &SerializeOptions::default())?;
    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }

    // Register the run-end kernel under FSST's id, so the wrong decoder runs on these bytes.
    let read_session = array_session();
    register_wasm_encodings(
        &read_session,
        [(
            "vortex.fsst".to_string(),
            ByteBuffer::from(RUNEND_KERNEL.to_vec()),
        )],
    )?;

    let result = SerializedArray::try_from(concat.freeze())?.decode(
        &dtype,
        len,
        &ReadContext::new(array_ctx.to_ids()),
        &read_session,
    );
    assert!(result.is_err(), "a mismatched kernel must not succeed");
    Ok(())
}

#[test]
fn native_encoding_supersedes_wasm_kernel() -> VortexResult<()> {
    // With the native encoding present, the kernel must NOT be registered.
    let session = native_session();
    let registered = register_wasm_encodings(
        &session,
        [(
            "fastlanes.bitpacked".to_string(),
            ByteBuffer::from(BITPACKED_KERNEL.to_vec()),
        )],
    )?;
    assert!(registered.is_empty(), "native encoding must supersede");

    // Without it, the kernel registers.
    let bare = array_session();
    let registered = register_wasm_encodings(
        &bare,
        [(
            "fastlanes.bitpacked".to_string(),
            ByteBuffer::from(BITPACKED_KERNEL.to_vec()),
        )],
    )?;
    assert_eq!(registered, vec!["fastlanes.bitpacked".to_string()]);
    Ok(())
}
