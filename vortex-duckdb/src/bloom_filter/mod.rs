// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown of the bloom filters DuckDB hash joins build over their build side.
//!
//! When a hash join expects to probe more rows than it built, DuckDB builds a bloom filter over
//! the build-side join keys and pushes it into the probe-side scan as a table filter. Applying it
//! inside the Vortex scan means the rows it rules out are never decoded, exported, or fed to the
//! join.
//!
//! The filter is owned by the join hash table and is published after the build side finishes, so
//! [`BloomFilterContains`] reads it afresh on every batch rather than capturing it when the scan
//! is planned: until the join publishes the filter, every row passes.
//!
//! The `probe` module reimplements DuckDB's probe (`src/planner/filter/bloom_filter.cpp`) and the
//! `hash` module reimplements its hashing, so that Vortex and DuckDB rule out exactly the same
//! rows.

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::Columnar;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::BufferMut;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::mask::Mask;
use vortex::scalar::Scalar;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::session::ScalarFnSessionExt;
use vortex::session::VortexSession;
use vortex::session::registry::CachedId;

use crate::bloom_filter::hash::DuckDbHash;
use crate::bloom_filter::hash::NULL_HASH;
use crate::bloom_filter::hash::hash_bytes;
use crate::bloom_filter::probe::Sectors;
use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::BloomFilterData;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;

mod hash;
mod probe;
#[cfg(test)]
mod tests;

/// Registers the scalar functions this module needs into `session`.
pub fn register(session: &VortexSession) {
    session.scalar_fns().register(BloomFilterContains);
}

/// Probes the bloom filter a DuckDB hash join built over its build side.
///
/// Takes the join key column and returns, for every row, whether the filter might contain its
/// value. False positives are possible; false negatives are not, so a row the filter rejects
/// cannot join.
#[derive(Clone, Debug)]
pub struct BloomFilterContains;

impl ScalarFnVTable for BloomFilterContains {
    type Options = BloomFilterProbe;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("duckdb.bloom_filter.contains");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("keys"),
            _ => unreachable!("bloom_filter.contains has exactly one child"),
        }
    }

    /// The filter answers "might contain" for every row, including NULL keys, so the answer is
    /// never itself NULL.
    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        if key_kind(&args[0]) != Some(options.0.keys) {
            vortex_bail!(
                "bloom filter was built from {} keys, which does not match {}",
                options.0.keys,
                args[0]
            );
        }
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let row_count = args.row_count();
        let probe = &options.0;

        let Some(sectors) = probe.sectors() else {
            return Ok(all_rows_pass(row_count));
        };

        let keys = args.get(0)?.execute::<Columnar>(ctx)?;
        let passed = match keys {
            Columnar::Constant(constant) => {
                let passes = sectors.contains(hash_scalar(constant.scalar())?);
                probe.observe(usize::from(passes) * row_count, row_count);
                return Ok(ConstantArray::new(
                    Scalar::bool(passes, Nullability::NonNullable),
                    row_count,
                )
                .into_array());
            }
            Columnar::Canonical(canonical) => probe_canonical(&canonical, &sectors, ctx)?,
        };

        probe.observe(passed.true_count(), row_count);
        Ok(BoolArray::new(passed, Validity::NonNullable).into_array())
    }

    /// A NULL key does not propagate: DuckDB hashes it to a fixed value and probes the filter
    /// with that, which is how an equality join's filter rules NULL keys out.
    fn is_strict(&self, _options: &Self::Options) -> bool {
        false
    }

    /// Every key type [`BloomFilterContains::return_dtype`] accepts can be hashed and probed, so
    /// evaluating the function over values no row references is safe.
    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

fn all_rows_pass(row_count: usize) -> ArrayRef {
    ConstantArray::new(Scalar::bool(true, Nullability::NonNullable), row_count).into_array()
}

/// A verdict that every row shares.
fn uniform_verdict(len: usize, passes: bool) -> BitBuffer {
    if passes {
        BitBuffer::new_set(len)
    } else {
        BitBuffer::new_unset(len)
    }
}

/// The bloom filter a [`BloomFilterContains`] probes.
///
/// Equality and hashing compare the identity of the underlying filter rather than its contents,
/// so that an expression tree keeps a stable identity as the join fills the filter in.
#[derive(Clone)]
pub struct BloomFilterProbe(Arc<Probe>);

impl Debug for BloomFilterProbe {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomFilterProbe")
            .field("keys", &self.0.keys)
            .field("state", &format_args!("{self}"))
            .finish()
    }
}

impl BloomFilterProbe {
    /// Wraps the filter of a pushed-down `BFTableFilter`, which hashes keys of `keys` kind.
    fn new(data: BloomFilterData, keys: KeyKind) -> Self {
        Self(Arc::new(Probe {
            data,
            keys,
            rows_passed: AtomicU64::new(0),
            rows_probed: AtomicU64::new(0),
            paused: AtomicBool::new(false),
        }))
    }
}

#[cfg(test)]
impl BloomFilterProbe {
    /// The handle to the filter, so that tests can populate it the way a join would.
    fn data(&self) -> &BloomFilterData {
        &self.0.data
    }
}

impl PartialEq for BloomFilterProbe {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BloomFilterProbe {}

impl Hash for BloomFilterProbe {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

impl Display for BloomFilterProbe {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0.paused.load(Ordering::Relaxed) {
            write!(f, "paused")
        } else if self.0.data.sectors().is_none() {
            write!(f, "pending")
        } else {
            write!(f, "{}", self.0.keys)
        }
    }
}

/// How many rows a filter probes before its selectivity is judged, mirroring DuckDB's
/// `SelectivityOptionalFilter::BF_CHECK_N` vectors.
const SELECTIVITY_CHECK_ROWS: u64 = 75 * 2048;
/// The fraction of rows a filter may pass and still be worth probing, mirroring DuckDB's
/// `SelectivityOptionalFilter::BF_THRESHOLD`.
const SELECTIVITY_THRESHOLD: f64 = 0.25;

struct Probe {
    data: BloomFilterData,
    keys: KeyKind,
    rows_passed: AtomicU64,
    rows_probed: AtomicU64,
    /// Set once the filter has proven too weak to be worth probing.
    paused: AtomicBool,
}

impl Probe {
    /// The filter's bits, or `None` while it is not worth probing: either the join has not
    /// published the filter yet, or the filter turned out not to be selective.
    fn sectors(&self) -> Option<Sectors<'_>> {
        if self.paused.load(Ordering::Relaxed) {
            return None;
        }
        self.data.sectors().map(Sectors::new)
    }

    /// Records how many of `probed` rows the filter let through, and gives up on a filter that
    /// rules out too few of them to pay for itself. DuckDB judges the bloom filters it evaluates
    /// itself the same way, once it has seen enough rows.
    fn observe(&self, passed: usize, probed: usize) {
        let probed = probed as u64;
        let probed_before = self.rows_probed.fetch_add(probed, Ordering::Relaxed);
        let total_passed =
            self.rows_passed.fetch_add(passed as u64, Ordering::Relaxed) + passed as u64;

        let total_probed = probed_before + probed;
        let crossed_check_point =
            probed_before < SELECTIVITY_CHECK_ROWS && total_probed >= SELECTIVITY_CHECK_ROWS;
        if crossed_check_point && total_passed as f64 >= total_probed as f64 * SELECTIVITY_THRESHOLD
        {
            self.paused.store(true, Ordering::Relaxed);
        }
    }
}

/// The physical representation DuckDB hashes a join key as.
///
/// DuckDB hashes by physical type, so this is what a Vortex column has to agree with for the two
/// to derive the same hash from the same value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeyKind {
    Bool,
    Primitive(PType),
    Bytes,
}

impl Display for KeyKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool => write!(f, "bool"),
            Self::Primitive(ptype) => write!(f, "{ptype}"),
            Self::Bytes => write!(f, "bytes"),
        }
    }
}

/// The way DuckDB hashes values of `dtype`, or `None` when Vortex cannot reproduce it.
///
/// Extension types hash as their storage, which is what DuckDB does for the logical types they
/// map onto: a DATE hashes as its `int32_t` day count and a TIMESTAMP as its `int64_t`.
fn key_kind(dtype: &DType) -> Option<KeyKind> {
    match dtype {
        DType::Extension(ext) => key_kind(ext.storage_dtype()),
        DType::Bool(_) => Some(KeyKind::Bool),
        DType::Primitive(PType::F16, _) => None,
        DType::Primitive(ptype, _) => Some(KeyKind::Primitive(*ptype)),
        DType::Utf8(_) | DType::Binary(_) => Some(KeyKind::Bytes),
        _ => None,
    }
}

/// The physical representation DuckDB hashes values of a logical type as.
///
/// Only types whose Vortex representation is bit-for-bit what DuckDB hashes are listed. In
/// particular UUID is left out: DuckDB hashes it as a `uhugeint_t` with a flipped sign bit, not
/// as the bytes Vortex stores.
fn duckdb_key_kind(key_type: &LogicalTypeRef) -> Option<KeyKind> {
    Some(match key_type.as_type_id() {
        DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => KeyKind::Bool,
        DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => KeyKind::Primitive(PType::I8),
        DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => KeyKind::Primitive(PType::I16),
        DUCKDB_TYPE::DUCKDB_TYPE_INTEGER | DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
            KeyKind::Primitive(PType::I32)
        }
        DUCKDB_TYPE::DUCKDB_TYPE_BIGINT
        | DUCKDB_TYPE::DUCKDB_TYPE_TIME
        | DUCKDB_TYPE::DUCKDB_TYPE_TIME_NS
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS
        | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ => KeyKind::Primitive(PType::I64),
        DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => KeyKind::Primitive(PType::U8),
        DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => KeyKind::Primitive(PType::U16),
        DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => KeyKind::Primitive(PType::U32),
        DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => KeyKind::Primitive(PType::U64),
        DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => KeyKind::Primitive(PType::F32),
        DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => KeyKind::Primitive(PType::F64),
        DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR | DUCKDB_TYPE::DUCKDB_TYPE_BLOB => KeyKind::Bytes,
        _ => return None,
    })
}

/// Builds the probe for a bloom filter pushed onto a column of `dtype`.
///
/// Returns `None` when the filter cannot be applied in Vortex, either because Vortex does not
/// reproduce DuckDB's hash for the key type or because the column is not what DuckDB believes the
/// join key to be. The bloom filter is always pushed inside an optional filter, so declining it
/// only costs the rows it would have ruled out.
pub fn try_new_probe(
    data: BloomFilterData,
    key_type: &LogicalTypeRef,
    dtype: &DType,
) -> Option<BloomFilterProbe> {
    let keys = duckdb_key_kind(key_type)?;
    if key_kind(dtype) != Some(keys) {
        return None;
    }
    // The column DuckDB scans must be the type it built the filter from, or the values it hashed
    // are not the values Vortex would hash.
    if LogicalType::try_from(dtype).ok()?.as_type_id() != key_type.as_type_id() {
        return None;
    }
    Some(BloomFilterProbe::new(data, keys))
}

fn probe_canonical(
    canonical: &Canonical,
    sectors: &Sectors<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    match canonical {
        Canonical::Bool(array) => probe_bool(array, sectors, ctx),
        Canonical::Primitive(array) => probe_primitive(array, sectors, ctx),
        Canonical::VarBinView(array) => probe_varbinview(array, sectors, ctx),
        Canonical::Extension(array) => probe_extension(array, sectors, ctx),
        Canonical::Null(array) => Ok(uniform_verdict(array.len(), sectors.contains(NULL_HASH))),
        other => vortex_bail!("unsupported bloom filter key type: {}", other.dtype()),
    }
}

fn probe_extension(
    array: &ExtensionArray,
    sectors: &Sectors<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    let storage = array.storage_array().clone().execute::<Canonical>(ctx)?;
    probe_canonical(&storage, sectors, ctx)
}

fn probe_bool(
    array: &BoolArray,
    sectors: &Sectors<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    // A boolean column has only two keys to hash, whatever its length.
    let key_hashes = [false.duckdb_hash(), true.duckdb_hash()];
    let values = array.bit_buffer_view();
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;

    Ok(probe_rows(array.len(), &validity, sectors, |idx| {
        key_hashes[usize::from(values.value(idx))]
    }))
}

fn probe_primitive(
    array: &PrimitiveArray,
    sectors: &Sectors<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;

    macro_rules! probe_slice {
        ($ty:ty) => {{
            let values = array.as_slice::<$ty>();
            probe_rows(array.len(), &validity, sectors, |idx| {
                values[idx].duckdb_hash()
            })
        }};
    }

    Ok(match array.ptype() {
        PType::I8 => probe_slice!(i8),
        PType::I16 => probe_slice!(i16),
        PType::I32 => probe_slice!(i32),
        PType::I64 => probe_slice!(i64),
        PType::U8 => probe_slice!(u8),
        PType::U16 => probe_slice!(u16),
        PType::U32 => probe_slice!(u32),
        PType::U64 => probe_slice!(u64),
        PType::F32 => probe_slice!(f32),
        PType::F64 => probe_slice!(f64),
        PType::F16 => vortex_bail!("DuckDB has no half-precision type to hash"),
    })
}

fn probe_varbinview(
    array: &VarBinViewArray,
    sectors: &Sectors<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    let buffers: Vec<&Buffer<u8>> = array.data_buffers().iter().map(|b| b.as_host()).collect();
    let views = array.views();

    let hash_view = |view: &BinaryView| {
        if view.is_inlined() {
            hash_bytes(view.as_inlined().value())
        } else {
            let view = view.as_view();
            hash_bytes(&buffers[view.buffer_index as usize][view.as_range()])
        }
    };

    Ok(probe_rows(array.len(), &validity, sectors, |idx| {
        hash_view(&views[idx])
    }))
}

/// How many rows are hashed before any of them is probed.
///
/// One word of output, and enough hashes in flight that their sector loads overlap.
const PROBE_CHUNK: usize = 64;

/// Probes the filter for every row, hashing an invalid row to the value DuckDB assigns a NULL
/// key, which is how DuckDB itself probes one.
///
/// Each sector load is a random access into a filter that is far larger than L1, so the rows are
/// hashed a chunk at a time and their sectors prefetched together. Waiting for one load before
/// starting the next costs around half the probe's time.
fn probe_rows(
    len: usize,
    validity: &Mask,
    sectors: &Sectors<'_>,
    hash_row: impl Fn(usize) -> u64,
) -> BitBuffer {
    match validity {
        Mask::AllTrue(_) => probe_hashes(len, sectors, hash_row),
        Mask::AllFalse(_) => uniform_verdict(len, sectors.contains(NULL_HASH)),
        Mask::Values(values) => {
            let valid = values.bit_buffer();
            probe_hashes(len, sectors, |idx| {
                if valid.value(idx) {
                    hash_row(idx)
                } else {
                    NULL_HASH
                }
            })
        }
    }
}

fn probe_hashes(len: usize, sectors: &Sectors<'_>, hash_row: impl Fn(usize) -> u64) -> BitBuffer {
    let mut passed = BufferMut::<u8>::with_capacity(len.div_ceil(PROBE_CHUNK) * 8);
    let mut hashes = [0u64; PROBE_CHUNK];

    for start in (0..len).step_by(PROBE_CHUNK) {
        let chunk = PROBE_CHUNK.min(len - start);

        for (offset, hash) in hashes[..chunk].iter_mut().enumerate() {
            *hash = hash_row(start + offset);
        }
        for hash in &hashes[..chunk] {
            sectors.prefetch(*hash);
        }

        let mut word = 0u64;
        for (bit, hash) in hashes[..chunk].iter().enumerate() {
            word |= u64::from(sectors.contains(*hash)) << bit;
        }
        // The tail writes a whole word too; the bits past `len` are zero and never read.
        passed.extend_from_slice(&word.to_le_bytes());
    }

    BitBuffer::new(passed.freeze(), len)
}

/// The hash DuckDB would compute for a scalar, matching [`DuckDbHash`] for the array kernels.
fn hash_scalar(scalar: &Scalar) -> VortexResult<u64> {
    if scalar.is_null() {
        return Ok(NULL_HASH);
    }
    Ok(match scalar.dtype() {
        DType::Extension(_) => hash_scalar(&scalar.as_extension().to_storage_scalar())?,
        DType::Bool(_) => scalar
            .as_bool()
            .value()
            .vortex_expect("non-null boolean value")
            .duckdb_hash(),
        DType::Primitive(ptype, _) => {
            let primitive = scalar.as_primitive();
            macro_rules! hash_typed {
                ($ty:ty) => {
                    primitive
                        .typed_value::<$ty>()
                        .vortex_expect("non-null primitive value")
                        .duckdb_hash()
                };
            }
            match ptype {
                PType::I8 => hash_typed!(i8),
                PType::I16 => hash_typed!(i16),
                PType::I32 => hash_typed!(i32),
                PType::I64 => hash_typed!(i64),
                PType::U8 => hash_typed!(u8),
                PType::U16 => hash_typed!(u16),
                PType::U32 => hash_typed!(u32),
                PType::U64 => hash_typed!(u64),
                PType::F32 => hash_typed!(f32),
                PType::F64 => hash_typed!(f64),
                PType::F16 => vortex_bail!("DuckDB has no half-precision type to hash"),
            }
        }
        DType::Utf8(_) => hash_bytes(
            scalar
                .as_utf8()
                .value()
                .vortex_expect("non-null utf8 value")
                .as_bytes(),
        ),
        DType::Binary(_) => hash_bytes(
            scalar
                .as_binary()
                .value()
                .vortex_expect("non-null binary value")
                .as_slice(),
        ),
        other => vortex_bail!("unsupported bloom filter key type: {other}"),
    })
}
