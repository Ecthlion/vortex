<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# WASM encodings in the Vortex file format

Status: **draft / in-progress implementation**

## Motivation

Vortex encodings are compiled into the reader. Adding a new encoding means shipping a new
release of every reader (Rust, Python, Java, the WASM web reader, DuckDB/DataFusion
integrations, ...). This makes it expensive to:

- experiment with new compression schemes,
- ship dataset-specific or domain-specific encodings,
- read files written by a *newer* writer than the reader.

The goal of this work is to embed the *decoder* for an encoding **inside the file** as a
sandboxed WebAssembly module. A reader that lacks the native encoding decodes the same bytes by
running the embedded kernel.

## Architecture

**The data is a normal serialized array; the kernel is only a fallback decoder.**

An encoding with a wasm kernel is an ordinary Vortex array encoding, written in the existing
serialized format (the `ArrayNode` flatbuffer, its metadata, buffers, and children). Nothing about
the data changes. The file additionally carries the compiled `.wasm` decoder blobs, and at read
time they are **merged into the session's encoding registry**:

1. The reader collects the file's embedded kernels (`(encoding_id, wasm bytes)` pairs).
2. [`register_wasm_encodings`] skips every id the session already knows — **a native encoding
   always supersedes its kernel** — and registers a [`WasmEncodingPlugin`] for the rest.
3. Deserialization proceeds exactly as before. When the serde layer resolves an unknown encoding
   id, it now finds the wasm plugin, whose `deserialize` receives the node's real serialized
   parts — the same `(dtype, len, metadata, buffers, children)` a native `VTable::deserialize`
   gets — runs the kernel, and returns the **decoded** array.

Consequences:

- A reader **with** the native encoding never touches the kernel and pays nothing.
- A reader **without** it decodes the same bytes through the sandbox, transparently — no special
  layout, no reader changes beyond registering the kernels at file-open.
- Wasm-backed encodings are **decode-only**: the plugin returns canonical data, so nothing
  wasm-specific survives past deserialization, and untrusted file code never touches the
  query-planning path. There is deliberately no pushdown.

### Kernels live alongside their encodings

Each kernel is a small `cdylib` crate next to the native encoding it mirrors, sharing the same
pure decode libraries so semantics match by construction:

| Kernel crate | Encoding | Reuses |
| --- | --- | --- |
| `encodings/fastlanes/wasm` | `fastlanes.bitpacked` | the [`fastlanes`] crate's unpack kernels |
| `encodings/runend/wasm` | `vortex.runend` | nothing — the SDK's generic `take` over its values child (18 KB) |
| `encodings/fsst/wasm` | `vortex.fsst` | [`fsst`] (fsst-rs)'s `Decompressor` |
| `encodings/onpair/wasm` | `vortex.onpair` | the [`onpair`] crate's `try_decode_into` |

The kernel crates are workspace-excluded (they carry their own size-optimized release profiles
and build standalone for `wasm32-unknown-unknown`); the parity contract with the native encoding
is enforced by round-trip tests that serialize with the **native** encoder and decode through the
kernel (`vortex-wasm/tests/plugin_roundtrip.rs`).

### File format: kernels as postscript-referenced segments

Kernels are ordinary segments written just before the other footer segments, and referenced from
the **postscript** by encoding id:

```
table Postscript {
    dtype: PostscriptSegment;
    layout: PostscriptSegment;
    statistics: PostscriptSegment;
    footer: PostscriptSegment;
    // Embedded decoder kernels, one per encoding id.
    wasm_kernels: [WasmKernelSpec];
}

table WasmKernelSpec {
    /// The array encoding id this kernel decodes (e.g. "fastlanes.bitpacked").
    id: string (required);
    /// ABI version the kernel was built against.
    abi_version: uint32;
    /// Location of the `.wasm` blob.
    segment: PostscriptSegment (required);
}
```

Writing: `VortexWriteOptions::with_wasm_kernel(EmbeddedKernel)`. `vortex_wasm::embed_kernel` builds
one, compiling the module first so a kernel this host could not run is rejected at write time
rather than by whoever reads the file. Attaching a kernel does not change how the data is encoded —
the writer still needs the native encoding to produce it.

Reading: `FooterDeserializer` extends its read window over the kernel segments, slices out the
blobs, and hands them to an `EmbeddedKernelLoader` before anything resolves an encoding id. Three
properties fall out of where the id lives:

- **The id is in the postscript, not the kernel segment**, so the reader can decide whether it
  wants the bytes at all. Kernels for encodings it can already decode are never fetched — a native
  reader pays for the postscript entry only. (`a_native_reader_does_not_read_the_kernel_segment`
  measures this differentially: same file, two sessions, one reads a megabyte more than the other.)
- **Running file-supplied code is opt-in.** `vortex-file` knows nothing about wasm; without
  `vortex_wasm::with_wasm_kernel_loader` installed on the session, embedded kernels are ignored and
  an unknown encoding fails exactly as it does today. This also keeps `wasmtime` out of
  `vortex-file`.
- **A file's kernels are scoped to that file.** The loader forks the array registry rather than
  registering into the caller's session, so two files using the same encoding id cannot end up
  decoded by each other's code. (Cloning a session or an `ArraySession` shares the underlying
  map; `VortexSession::fork` and `ArraySession::fork` are the independent copies this needs.)

The declared `abi_version` is checked before the module is compiled, and the module's own
`vx_abi_version` export is checked before it is run — the first catches a stale kernel cheaply, the
second catches a postscript that lies about one.

Kernels are naturally content-addressable for dedup across files; that, and caching a compiled
kernel across the files that share it, are not yet implemented.

## Crates

- **`vortex-wasm` (the host)** — depends on `vortex-array`, `vortex-session`, `vortex-file`, and
  `wasmtime`. Provides [`WasmKernel`]/`WasmDecoder` (the runtime), `convert` (the boundary),
  [`WasmEncodingPlugin`] (the `ArrayPlugin` adapter), [`register_wasm_encodings`], and
  [`WasmKernelLoader`]/`with_wasm_kernel_loader`/`embed_kernel` (the file-format wiring).
- **`vortex-wasm-guest` (the guest SDK)** — `#![no_std]`, dependency-free (`core`/`alloc`).
  Provides the ABI (`abi`), the frame views (`node`), the array buffer builder/reader
  (`data`), a tiny protobuf reader for prost metadata (`proto`), the bump-allocator runtime
  (default `runtime` feature; disable it when a dependency links `std`), and the
  [`WasmEncoding`] trait + `export_wasm_encoding!` macro.

## One result shape: the kernel always materializes

A kernel's output is **a canonical array of the node's own dtype**, whatever that dtype is. The
host hands every declared child to the kernel in Vortex's canonical layout and reads one canonical
array back; there is no second result kind.

Two kinds of encoding still exist, and the difference matters for how a kernel is *written*, not
for what it returns:

- **Value-producing** (bit-packing, FSST, OnPair, zstd, delta, ALP): the output bytes exist nowhere
  until the kernel computes them. The kernel reads its buffers and writes elements.
- **Re-arranging** (run-end, dict, sparse, masked, extension-over-storage): the output is a child's
  values reordered or overlaid. The kernel computes indices and applies the SDK's generic
  [`Decoded::take`](../../vortex-wasm-guest/src/data.rs) to the canonical child, which follows the
  canonical layout of *any* dtype — strings, decimals, lists, structs, unions, maps, extension
  types — so the kernel needs no per-dtype code and is generic over types added after it was built.

An earlier revision of this design let a re-arranging kernel return a *plan* (`take(child(1),
indices)`) that the host evaluated lazily over children it never canonicalized. That kept the
values out of the sandbox but split the decode across two trust domains, gave the host a small
interpreter to validate, and made a kernel's semantics depend on which host constructors existed.
The design now trades that laziness for a single, closed contract: **the kernel owns the whole
decode, and the host only has to check that what came back is a well-formed array of the promised
type.** [The array frame](#the-array-frame) below is that contract; the cost of the trade is
measured in [the run-end section](#vortexrunend-encodingsrunendwasm--the-structural-case).

## The encoding trait (guest)

A kernel is the portable mirror of a native `VTable::deserialize`. Because only the encoding
knows its children's dtypes, decoding is two-phase:

```rust
pub trait WasmEncoding {
    /// From the metadata (and the serialized child count), declare each child's dtype and length.
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>>;

    /// Decode the node into its canonical array, of the node's own dtype and length.
    fn decode(node: &NodeView<'_>) -> GuestResult<Decoded>;
}

export_wasm_encoding!(MyEncoding); // defines vx_abi_version + vx_alloc + vx_children + vx_decode
```

`ChildSpec::new(dtype, len)` declares a serialized child. The host decodes it in its own encoding
(natively, or recursively through another kernel), canonicalizes it, and copies it into guest
memory as an [`ArrayView`](#the-array-frame). The dtype is a [`DTypeExpr`](#the-dtype-channel): a
literal, or a derivation such as `DTypeExpr::parent()`, which is all a re-arranging kernel needs
since its values child has its own type.

`NodeView` exposes the node's full dtype, the metadata bytes (parse with `proto`), the raw buffers
(resident in guest memory, 8-byte aligned so they can be cast in place), and each child as an
`ArrayView` — a borrowed, recursive view whose `kind` is one variant per canonical shape.
`Decoded` is the owned mirror of that view; a kernel either builds one from its buffers or asks
`Decoded::take(&child, &indices)` to gather one.

## Host / guest ABI (`abi_version = 1`)

All integers little-endian; the single linear memory is exported as `"memory"`. The ABI is
**push-based**: there are no host callbacks during decode.

Guest exports:

- `vx_abi_version() -> u32` — the ABI the kernel was built against. The host reads it at compile
  time and refuses a kernel that disagrees with its own, so a stale kernel fails loudly instead of
  misreading frames.
- `vx_alloc(len) -> ptr` — bump allocation; the host uses it to place all inputs.
- `vx_children(frame_ptr, frame_len) -> ptr` — input
  `[u64 len][u32 flags][u32 n_children][u32 dtype_len][u32 metadata_len][dtype][metadata]`; output
  `[u32 n]` + `n` descriptors `[u32 dtype_len][u64 len][dtype]`.
- `vx_decode(frame_ptr, frame_len) -> ptr` — input
  `[u64 len][u32 flags][u32 dtype_len][u32 metadata_len][u32 n_buffers][u32 n_children]`
  `[dtype][metadata][(ptr,len) x buffers][u32 child_ptr x children]`, where each child pointer
  addresses an [array frame](#the-array-frame) the host wrote; output points at one array frame —
  the node's canonical array.

`flags` bit 0 is the parent's nullability, kept because it is free and it is what most kernels
branch on; the full type travels as a real [dtype expression](#the-dtype-channel). Negative
returns are error codes; panics become traps, which the host surfaces as decode errors.

### The array frame

Arrays cross in **both directions** in one recursive wire format that mirrors Vortex's `Canonical`
enum shape for shape. There is no schema in it: the host already holds the node's `DType`, the
guest declared its children's dtypes, and every frame is read *against* a dtype the reader trusts.

```text
frame = [u8 shape][u8 param][u8 validity][u8 pad][u32 len][u32 validity_ptr]
        [u32 n_buffers][(u32 ptr, u32 len) × n_buffers]
        [u32 n_children][frame × n_children]          -- inline, preorder
```

| `shape` | `param` | buffers | children | Vortex constructor on the way back |
| --- | --- | --- | --- | --- |
| `NULL` | | | | `NullArray::new` |
| `BOOL` | | bitmap | | `BoolArray::try_new` |
| `PRIMITIVE` | ptype | values | | `PrimitiveArray::from_byte_buffer` |
| `DECIMAL` | storage width `I8..I256` | values | | `DecimalArray::try_new_handle` |
| `VAR_BIN_VIEW` | | 16-byte views, then data buffers | | `VarBinViewArray::try_new` |
| `LIST` | | | offsets, sizes, elements | `ListViewArray::try_new` |
| `FIXED_SIZE_LIST` | | | elements | `FixedSizeListArray::try_new` |
| `STRUCT` | | | one per field | `StructArray::try_new` |
| `UNION` | | | type ids (`u8`, carries the outer validity), one per variant | `UnionArray::try_new` |
| `MAP` | | | entries (a `LIST` of `{key, value}` structs) | `MapArray::try_new` |
| `EXTENSION` | | | storage | `ExtensionArray::try_new` |

- `validity` is an **algebra** — `NonNullable | AllValid | AllInvalid | Bitmap` — so a non-nullable
  or all-valid array transmits no bitmap at all. Union, map, and extension frames carry no validity
  of their own: it lives in the type-ids, entries, and storage child respectively, as in Vortex.
- Strings cross as canonical views plus data buffers (which FSST and OnPair emit directly), lists
  as list-views, so a sublist is a slice of the elements and never copied. For primitives and bools
  the bytes are identical to Arrow's.
- Bitmaps are byte-aligned via `shrink_offset` before crossing, closing the bit-offset hazard a
  sliced array's mask would otherwise cause.
- **`Variant` is the one dtype with no frame.** Vortex defines no physical canonical layout for
  variant values (canonicalization is the identity; the storage is a constant or chunked array of
  variant scalars), so there is nothing to spell. A kernel may still *name* the type in the dtype
  channel; asking it to produce or consume variant *values* is rejected with a clear error.

The boundary *was* the Arrow C Data Interface. It was removed because Arrow C FFI is a
**schema-carrying protocol and this boundary has no schema to carry**, and because Arrow utf8's
i32 offsets import as `VarBin`, which is not canonical, so every string kernel paid a second full
conversion of the heap. What went with it: the `arrow-*` dependencies, ~800 lines of schema
recursion, metadata-blob parsing, and dictionary handling — and the attack surface they carried.

### What the host validates

A returned frame is untrusted file data. It is parsed with a depth limit (32), a node budget
(4096 frames), and a per-frame buffer cap (64), so a few hundred bytes of nested `LIST` tags cannot
overflow the host stack or drive an unbounded allocation. It is then *built* against the node's
dtype, level by level:

| check | where |
| --- | --- |
| `shape` matches the dtype at this level; `param` matches the ptype or a valid decimal width | `ArrayDescriptor::build` |
| every buffer lies inside guest memory; values buffers are exactly `len × width` bytes; bitmaps at least `ceil(len / 8)` | `copy_out`, `build` |
| a non-nullable dtype gets no validity; a validity bitmap is `len` bits | `validity` |
| child count matches the dtype (fields, variants, `3` for a list, `1` for FSL/map/extension) | `expect_children` |
| list offsets and sizes are non-nullable integers with one entry per list, and every `offset + size` lands inside the elements | `ListViewArray::try_new` |
| every string view's buffer index, offset, and size are in range, and utf8 is valid | `VarBinViewArray::try_new` |
| decimal storage is wide enough for the precision | `DecimalArray::try_new_handle` |
| union type ids are the variants' declared tags; map keys are non-nullable; extension storage matches | `UnionArray::try_new`, `MapArray::try_new`, `ExtensionArray::try_new` |
| every level's length and dtype equal what was asked for | `build`, then `WasmEncodingPlugin::deserialize` |

Guest memory dies with the instance, so each buffer is copied out once; the copy is where the
alignment Vortex wants (`ByteBuffer::copy_from_aligned`) is applied, so the guest writes plain
bytes and never has to know that a `DecimalArray<i128>` wants 16-byte alignment.

### Memory

Kernels are `#![no_std]`; the SDK provides a grow-only bump `#[global_allocator]` over linear
memory and a trap-on-panic handler behind the default `runtime` feature. A kernel instance
decodes exactly once and its whole memory is reclaimed when the host drops the per-decode store,
so `dealloc` is a no-op and there is no free in the ABI.

**`vx_alloc` returns 8-byte-aligned offsets — this is part of the ABI.** Every host upload (the
frames, the raw buffers, the child structs) lands aligned, so kernels view typed data **in
place**: wasm32 is little-endian, matching the serialized format, so e.g. the bitpacked kernel
casts its packed buffer to `&[u32]` (`align_to`, checked) instead of copying words out.

Today the bitpacked and FSST kernels disable the `runtime` feature because a dependency links `std`
(fastlanes' `num-traits` edge lacks `default-features = false`; fsst-rs is not `#![no_std]`) —
`std`'s dlmalloc/panic machinery then costs ~16 KB per blob. Both are one-PR fixes in
SpiralDB-owned crates; with the fastlanes fix applied locally, the identical kernel source builds
fully `no_std` (SDK runtime on) at **37 KB instead of 53 KB** and passes the whole round-trip
suite. Once upstreamed, kernels are no_std by default and the feature remains only as an escape
hatch.

## Worked kernels

### `fastlanes.bitpacked` (`encodings/fastlanes/wasm`)

Parses the real prost `BitPackedMetadata` (bit width, offset, optional `PatchesMetadata`) with
the SDK's proto reader; declares `[patch indices, patch values(parent dtype), (chunk offsets),
validity]` children; unpacks 1024-element FastLanes chunks with the **same [`fastlanes`] crate
kernels the native encoding uses**. The packed buffer is **cast, not copied** (`vx_alloc`'s
alignment guarantee + wasm32's little-endianness), and full in-range chunks unpack directly into
the output — mirroring the native `decode_into` fast path — with scratch only for a sliced first
chunk and a partial trailer. Patches overwrite `index - patches.offset` in the sandbox, exactly as
the native `apply_patches_to_uninit_range` does. The validity child carries through. Scope: 4-byte
primitives — other widths are pure monomorphization at ~25 KB of unrolled unpack code per width
family. Blob: **~58 KB** (the unpack kernels dominate; see [Binary size](#binary-size)).

This is the shape of a kernel that genuinely computes. It takes a buffer, reads it as typed words
in place, and writes elements:

```rust
let packed = node.buffer(0)?;                       // host-uploaded, 8-byte aligned
let (head, words, _) = unsafe { packed.align_to::<u32>() };
guest_ensure!(head.is_empty(), "packed buffer must be 4-byte aligned");

let mut out = alloc::vec![0u32; node.len];
for chunk in 0..num_chunks {
    let chunk_words = &words[chunk * words_per_chunk..(chunk + 1) * words_per_chunk];
    // fastlanes::BitPacking — the same kernels the native encoding calls
    unsafe { BitPacking::unchecked_unpack(bit_width, chunk_words, &mut out[dst..dst + CHUNK]) };
}

Ok(Decoded::Primitive(DecodedPrimitive { ptype, len: node.len, values, validity }))
```

There is no re-arrangement of an existing child that produces bit-unpacked output, so something
has to compute it. The kernel reads the ptype off `node.dtype()` — the buffer layout alone would
not say whether those four-byte values are `u32`, `i32`, or `f32` — and the host types the result
with the node's own dtype when it rebuilds the frame.

### `vortex.fsst` (`encodings/fsst/wasm`)

Parses `FSSTMetadata`; declares `[uncompressed_lengths, codes_offsets, validity]` children;
rebuilds the symbol table with `fsst::Symbol::from_slice` and bulk-decompresses the whole codes
heap with **the same [`fsst`] crate `Decompressor` the native canonical path uses**; the prefix
sums of the uncompressed lengths are exactly the output utf8 offsets. Blob: **~26 KB**.

### `vortex.onpair` (`encodings/onpair/wasm`)

OnPair is FSST-shaped — a trained dictionary in buffer 0, a stream of fixed-width codes indexing
it, per-row code boundaries, per-row uncompressed lengths — so its kernel is the same
*value-producing* shape and returns a canonical string array. Two things about it are worth
recording, because neither was true of the first two kernels.

**Every child ptype comes from the metadata.** The four integer children flow through the ordinary
cascading compressor, which narrows them: `codes` to U8 for a small dictionary, `dict_offsets` to
U16. The recorded ptype is the only thing that says how wide they are on disk, and the kernel widens
them back to the `u16`/`u32` the decoder wants. A kernel that assumed the natural widths would
misread a well-formed file.

**The decoder was already written for untrusted input — but it panics.** `onpair` validates the
dictionary (`CompactDictionaryView::validate`: offsets, token sizes, the trailing read padding)
before a view of it exists, and its decoders bounds-check every code. The checks are real; the
failure mode is a panic, which in a `panic = "abort"` guest reaches the host as an opaque trap. So
the kernel validates the dictionary through the fallible constructor and range-checks the codes
itself first, and a corrupt file becomes a clean kernel error. It also cross-checks the sum of
`uncompressed_lengths` against `decoded_len`, since those two independently describe the same
output and only agree if the file is honest. Blob: **~28 KB**.

The one wart is a dependency, not a design problem: `onpair` links `rand` for dictionary *training*,
which pulls `getrandom`, which refuses to build for `wasm32-unknown-unknown` without a backend. The
kernel crate selects the custom backend in `.cargo/config.toml` and supplies a shim that always
fails, since a decoder never draws randomness. A `train`/`compress` feature gate upstream would
remove it.

### `vortex.runend` (`encodings/runend/wasm`) — the structural case

Run-end is the canonical re-arranging encoding, and its kernel computes nothing about the values.
It declares `ends` (a primitive) and `values` (the parent's own dtype, whatever that is), expands
the run ends into one `u32` gather index per row — mirroring `trimmed_ends_iter`, so a sliced
array's `offset` is honoured — and hands them to `Decoded::take`:

```rust
fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>> {
    Ok(alloc::vec![
        ChildSpec::new(DTypeExpr::primitive(ends_ptype, false), meta.num_runs),
        // Same dtype as the parent, whatever that is: the kernel never needs to know.
        ChildSpec::new(DTypeExpr::parent(), meta.num_runs),
    ])
}

fn decode(node: &NodeView<'_>) -> GuestResult<Decoded> {
    let ends = node.child(ENDS)?.as_primitive()?;
    // ... expand run ends into one u32 run index per output row ...
    Decoded::take(&node.child(VALUES)?, &indices)   // any dtype; every index bounds-checked
}
```

`Decoded::take` is the SDK's one generic gather. It follows each canonical layout to its cheapest
reading: primitives and decimals are copied by width, bools by bit, string *views* are gathered
while the data buffers they point into are copied through once (no per-string copy), list offsets
and sizes are gathered while the elements pass through untouched, and structs, unions, maps, and
extension storage recurse. Validity is gathered alongside.

The consequences:

- **The kernel is dtype-agnostic.** The *native* decoder needs three separate implementations
  (bool / primitive / varbinview) and `vortex_bail!`s on anything else. This kernel has none —
  run-end over strings, and over a `struct { utf8?, list<i32>, timestamp? }?` the native decoder
  rejects outright, both work with zero dtype-specific code in the guest, and both are covered by
  tests.
- **Validity falls out.** Run-end's output validity is the values' validity gathered through the
  same runs; `take` reproduces that, so the kernel never touches validity.
- **It is 18 KB.** The generic gather is real code, monomorphized over the canonical shapes; the
  earlier lazy version that only *named* its child was 5.9 KB.

### What materializing costs

The previous revision of this kernel returned `take(child(VALUES), indices)` as a plan, and the
host evaluated it with `ArrayRef::take` over the child in its *own* encoding — a lazy `DictArray`,
no canonicalization, nothing copied in. That is what was given up:

| | delegating (previous) | materializing (shipped) |
| --- | --- | --- |
| Dtypes supported | every one, including nested | every one, including nested (Variant excepted) |
| Bytes crossing in | `num_runs` ends | `num_runs` ends **+ the canonical values child** |
| Bytes crossing out | `len` × 4 index bytes | `len` × element width, materialized |
| Host work | `take` → a lazy `DictArray` | copy and validate the guest's output |
| Host trust surface | a plan interpreter with 7 opcodes and its own validation table | one array-frame parser |
| Kernel semantics depend on | which constructors the host offers | nothing outside the kernel |
| Compiled size | 5.9 KB | 18 KB |

The delegating design kept the input out of the sandbox and left the output unmaterialized until
the scan asked for it; a filter that pruned the chunk paid for neither. Materializing pays both
costs up front. What it buys is a closed contract: the kernel is the *complete* decoder for its
encoding, the host does not need to grow a vocabulary (arithmetic, slicing, concatenation, ...)
to keep up with new encodings, and there is exactly one thing to validate on the way back. The
design accepts that trade; [laziness](#open-questions) is the open question it leaves.

## The dtype channel

The obvious cheap thing to send a guest is a coarse kind tag — primitive, bool, utf8, or *other*.
It does not work: anything outside the enumerated set is then fatal rather than slow.
`datetimeparts` needs a Timestamp's `TimeUnit` and timezone, `decimal` needs precision and scale,
`fixed_size_list` needs its size, and a kind tag shows none of it.

So the ABI sends a real type, in a compact preorder encoding with a tag byte per node
(`vortex-wasm-guest/src/dtype.rs` holds the grammar). **Every `DType` variant has a literal
spelling, in both directions**: null, bool, primitive, decimal (precision, scale), utf8, binary,
list, fixed-size list (size), struct (named fields), union (named variants with their type tags),
variant, map (key, value, `keys_sorted`), and extension.

An extension literal carries the id, the vtable's **own serialized metadata** (the same bytes the
file footer records — a Timestamp's unit and timezone, say), and the storage type. The host
rebuilds it exactly as the footer's dtype is rebuilt: the session's extension registry finds the
plugin for the id and asks it to deserialize the metadata against the storage type. An id the
session does not know becomes a `ForeignExtDType` placeholder if the session `allow_unknown()`s
foreign types, and is an error otherwise — the same policy as everywhere else in the reader, so a
kernel cannot smuggle in a type the reader would have refused from the footer. A kernel can
therefore both *inspect* an extension type it is handed (`node.dtype()?.extension_id()`,
`extension_metadata()`, `storage()`) and *declare* one for a child.

**Derivations** are the convenience on top. A kernel generic over its parent does not *want* to
name a concrete type, so a guest may instead write a path:

```text
Parent | Field(i, inner) | Element(inner) | Storage(inner) | Nullable(inner) | NonNullable(inner)
```

These compose — `NonNullable(Element(Field(1, Parent)))` is valid — and the host resolves them
against a `DType` it already trusts. That is how the run-end kernel stays dtype-agnostic: it
declares its values child as `Parent` and works over strings, decimals, structs, or any type added
later, with no code.

The host only ever writes literals, because it holds the real type and has nothing to derive from.
Both directions are bounded the same way: 32 levels of nesting, 4096 named entries per struct or
union, so a few hundred bytes of nested tags cannot overflow either stack.

## Editions and kernels: a new encoding, shipped with its decoder

Editions (`docs/specs/editions.md`) and embedded kernels answer different questions and compose.
An edition says which serialized ids a **writer** may emit, and which readers are guaranteed to
know them natively. A kernel says how a **reader** decodes an id it does not know. Nothing about
attaching a kernel changes what the writer is allowed to write, and nothing about an edition changes
whether a reader runs a kernel.

The editions spec uses `vortex.decimal_byte_parts_v2` as its motivating example — a wide-decimal
revision staged in a draft edition, which an older reader "reports as unknown instead of trying to
decode a wire format it does not support". Take that exact case and ask what it takes to ship it
with a kernel, layer by layer.

**The file and edition plumbing already work.** The writer enables the draft edition, so the
serialization context permits the `_v2` id; it attaches
`embed_kernel("vortex.decimal_byte_parts_v2", module)`. The kernel is keyed by the **wire id**,
which is exactly the id editions introduced, and the reader's `wanted_kernels` filter looks that id
up in the deserializer registry — the one keyed by wire ids. So the granularity is right by
construction: a reader whose native `DecimalByteParts` plugin predates `_v2` has `vortex.decimal_byte_parts`
registered and `_v2` not, and fetches only the `_v2` kernel; a reader with the new native plugin
fetches nothing. Kernels load before the layout is parsed, so they take precedence over the
`allow_unknown` placeholder path, and a kernel-decoded array is canonical, so nothing downstream
ever sees the `_v2` id. The one thing this changes about the editions guarantee is deliberate and
opt-in: a reader that installs the loader *does* decode a draft component it never shipped. A reader
without the loader behaves exactly as the spec says.

This also names a compatibility floor editions do not currently record. For a component shipped
with a kernel, the earliest reader is not the edition's `min_library_version` but **the release that
shipped the kernel loader and this ABI** — the same floor for every such component, forever, which
is the point of the exercise.

**Reading the inputs works.** `vortex.decimal_byte_parts` has one child, `msp`, a signed
primitive whose ptype is in the metadata; `_v2` adds `lower_part_count` unsigned 64-bit children.
Every input is a primitive, so `ChildSpec::new(DTypeExpr::primitive(..))` covers all of them, and
`node.dtype()?.kind()` hands the kernel `Decimal(precision, scale)` so it knows the target width.

**Producing the output works.** `Decoded::Decimal` carries one values buffer plus a storage-width
tag (`I8`..`I256`); the host rebuilds it with `DecimalArray::try_new_handle`, which also checks the
storage is wide enough for the precision. The guest writes plain bytes; the 16/32-byte alignment
Vortex wants is applied by the copy every result already takes.

- For the **v1** form, `to_canonical_decimal` *reinterprets* the signed primitive's buffer as
  decimal storage of the same width, computing nothing. The kernel does the same: it takes the
  `msp` child's bytes and re-emits them under the decimal shape with the child's validity. One
  copy, no arithmetic, generic over every storage width.
- For **`_v2`**, combining parts is arithmetic — `(msp as i128) << 64 | lower` — so it must be
  computed. `i128` is native on wasm32; `i256` is two-limb arithmetic the guest writes itself,
  since it cannot link the host's type.

So: the kernel model handles the new encoding, the editions model handles who may write it, and the
two meet at the wire id. Nothing in the ABI stands between the example and a passing round-trip;
what remains is writing the kernel.

## Remaining ABI gaps

- **`Variant` values cannot cross.** Every other dtype has an array frame; Vortex defines no
  physical canonical layout for variant values, so there is nothing for a kernel to read or write.
  The dtype channel still spells the type, and the host rejects a variant *array* in either
  direction with a clear error rather than inventing a layout the rest of Vortex does not have.
- **Every declared child is canonicalized and copied in.** A re-arranging kernel over a wide
  values child pays for the whole child even if the runs it expands reference a fraction of it, and
  `vortex.chunked` — whose children *are* the output — would copy every chunk into the sandbox and
  back. The earlier plan-based design avoided this by letting the host gather lazily; the
  materializing design accepts it for a closed contract (see
  [What materializing costs](#what-materializing-costs)).
- **`vortex.chunked` is inexpressible at any cost**: its per-chunk lengths are the *decoded
  contents of child 0*, and `children` is a single pure call with a mandatory length. Fixing it
  requires an iterative declaration phase.
- **`children` cannot request a row window of a child.** `ChildSpec` carries a dtype and a length,
  so a kernel that needs only part of a child still gets all of it. `vortex.onpair` is the live
  case: `codes_offsets` bounds the run of `codes` belonging to the rows present, and the native
  path point-looks-up those two boundaries and slices `codes` before materializing it. The kernel
  slices the same window, but only after the host has already decoded and copied the whole child
  in. Measured on a 161-of-400-row slice, the serialized array is 78% of the full array's bytes
  rather than 40%, so the over-read is most of the codes stream. This is the `chunked` problem in a
  milder form — the bound lives inside another child, which the single pure `vx_children` call
  cannot read — and it wants the same fix: a second declaration round, or a `ChildSpec` row range
  the host applies with `slice` before canonicalizing.
- **No laziness.** The plugin decodes eagerly at deserialize time, and the kernel's output is
  fully materialized. A filter that prunes a chunk still pays for decoding it.

## Runtime choice: `wasmtime`

The host embeds [`wasmtime`](https://wasmtime.dev) (pinned to the **36.x LTS** line, 24-month
support). It was chosen over `wasmer` and the earlier `wasmi` because the kernels are **untrusted
file data on a decode hot path**, and `wasmtime`:

- is the Bytecode Alliance reference runtime with the strongest sandboxing track record
  (continuous OSS-Fuzz with differential oracles, a formal CVE process);
- has first-class facilities for bounding untrusted code — `StoreLimits` (memory/instance caps),
  plus fuel and epoch interruption for CPU time;
- is built for *many short-lived instances* (pooling allocator, copy-on-write memory init,
  `InstancePre`) — exactly the decode pattern;
- exposes an API `wasmi` deliberately mirrors, so the host code is nearly identical either way.

We use the default **Cranelift** backend (not the newer Winch/Pulley, which are less
battle-tested) and compile each kernel once, instantiating a fresh `Store` per node decode.

> **`wasm32` / browser caveat.** Neither `wasmtime` nor `wasmer` can execute guest wasm while the
> runtime *itself* is compiled to `wasm32-unknown-unknown` (wasmtime's `runtime` feature does not
> build for wasm32; wasmer only delegates to the host's `WebAssembly` engine). Only `wasmi`
> (a pure-Rust interpreter) self-hosts in wasm32. Vortex does target `wasm32-unknown-unknown`
> (the `wasm-test` crate, `vortex-web`), but `vortex-wasm` is not in that build today. If the
> browser reader ever needs WASM encodings, the clean path is selecting `wasmi` behind
> `#[cfg(target_arch = "wasm32")]` — its API mirrors `wasmtime`.

### Sandboxing & resource limits

`wasmtime` is a sandbox: no host memory access beyond the explicit imports, no syscalls. We
additionally cap guest linear-memory growth per decode via `StoreLimits`, cap declared
child/buffer counts and type/frame recursion depth, rebuild every guest-returned frame through
Vortex's own checked constructors (see [What the host validates](#what-the-host-validates)), and
treat any guest trap as a decode error (never a host panic). CPU-time bounding (wasmtime fuel or epoch interruption) is a planned follow-up. The
kernel is untrusted data from the file, exactly like array bytes; a buggy kernel can only corrupt
*that array's* values, never host memory.

## Binary size

Compiled `wasm32-unknown-unknown`, size-optimized (`opt-level = "z"`, `lto`, `panic = "abort"`,
`strip`):

| kernel | size | notes |
|---|---|---|
| minimal SDK kernel (no_std, no deps) | ~4 KB | the SDK floor: allocator + buffer glue |
| `vortex.runend` | ~18 KB | the generic `Decoded::take` over every canonical shape; no decode library |
| `vortex.onpair` | ~32 KB | the `onpair` crate's decoder |
| `vortex.fsst` | ~33 KB | fsst-rs `Decompressor` + `std` (fsst-rs is not yet no_std) |
| `fastlanes.bitpacked` | ~58 KB | fastlanes unrolled unpack kernels + `std` via num-traits |

The recursive array frame and the generic gather cost every kernel a few KB over the earlier
plan-returning SDK (bitpacked was 51 KB, fsst 26 KB); run-end tripled, because it now carries the
gather instead of naming its child.

The early prototype showed why the SDK avoids Vortex crates entirely: pulling `vortex-error`
(which drags `jiff`/`prost`/`arrow-schema`) put kernels at ~74 KB before any real decode logic.
Kernels are read once per file and cached, so tens of KB is acceptable; the `std` relinks are
fixable upstream (`num-traits` default features in fastlanes-rs).

## Implementation phases

1. **Prototype (done, superseded):** `WasmLayout` + payload/child write model over `wasmi`, then
   `wasmtime`; proved the VM, the boundary, and end-to-end round trips.
2. **Arrow C Data Interface, then removed (done):** the boundary was briefly a complete, generic
   Arrow C FFI binding. It was deleted once it became clear Arrow is a schema-carrying protocol
   and this boundary carries no schema — see
   [the array frame](#the-array-frame).
3. **Session-level wasm encodings (done):** `WasmLayout` removed. Kernels decode the **real
   serialized parts** (`vx_children` plus the pushed `vx_decode` frame);
   [`WasmEncodingPlugin`] registers under the encoding's id and returns decoded arrays;
   [`register_wasm_encodings`] merges kernels into a session with native-supersedes semantics.
   Kernels live alongside their encodings (`encodings/fastlanes/wasm`, `encodings/fsst/wasm`)
   and reuse the same decode crates; parity is tested against natively-serialized bytes,
   including patches and nullable columns.
4. **Structural decoding via a host-evaluated `Take` (done, superseded by 9):** the survey found
   that ~20 of ~29 encodings only re-arrange a child. A per-child access mode let a kernel name a
   child without the host canonicalizing or copying it, and the host performed the gather with
   `ArrayRef::take` (a lazy `DictArray`). Proven by the `vortex.runend` kernel over strings.
5. **Untrusted-input hardening (done):** `SerializedArray::decode` and the `ArrayChildren` blanket
   impl now return `VortexError` instead of `assert!`-ing, so a lying kernel cannot abort the
   process.
6. **File plumbing (done):** the postscript `wasm_kernels` field, `with_wasm_kernel` on the
   writer, and loader-based registration at file-open — opt-in, file-scoped, and fetching only the
   kernels the reader actually lacks (see above). A `vx_abi_version` guest export makes a stale
   kernel a clear error rather than a misread frame.
7. **Plan vocabulary and dtype channel (done, plan superseded by 9):** `vx_decode` returned a
   flat postorder plan — `Materialized`, `Child`, `Take`, `Slice`, `Concat`, `Constant`,
   `SetValidity` — each opcode one `vortex-array` constructor, evaluated in a single non-recursive
   forward pass with per-node validation and an output budget. Types cross in the full
   [dtype channel](#the-dtype-channel), with derivations.
8. **A fourth kernel, `vortex.onpair` (done):** the first kernel written *against* the settled ABI
   rather than alongside it, and it needed no ABI change — FSST's shape. What it did surface is the
   child-windowing gap above, plus a dependency wart: `onpair` links `rand` for training, so the
   kernel crate has to select `getrandom`'s custom backend to build for wasm32 at all.
9. **Always materialize, every dtype (done):** the plan is gone. A kernel returns one canonical
   [array frame](#the-array-frame) of the node's own dtype, and receives its children the same way,
   with a frame shape for every `Canonical` variant — null, bool, primitive, decimal, string views,
   list, fixed-size list, struct, union, map, extension — and `Variant` rejected explicitly. The
   guest SDK's `Decoded::take` is the generic gather re-arranging kernels use in-sandbox. The dtype
   channel gained extension literals with real metadata, resolved through the session registry in
   both directions. Proven by `vortex.runend` over a nested struct-of-list-of-timestamp dtype the
   native decoder cannot canonicalize.
10. **Breadth (next):** more kernels (dict, ALP, sparse, `decimal_byte_parts`), kernel dedup +
    cross-file caching, CPU-time limits, and the `wasm32` fallback runtime for the browser reader.

Pushdown (filter/pruning into the kernel) is explicitly **out of scope** — WASM encodings only
decompress; the engine filters on the decoded output.

## Open questions

- **Laziness:** the plugin decodes eagerly at deserialize time and the kernel materializes its
  whole output. A lazy wrapper array (decode on first execute) would let filters skip decodes for
  pruned ranges; a host-side gather for re-arranging kernels would keep wide children out of the
  sandbox. Both were traded away for the closed contract above and could return behind it.
- **Kernel caching key:** blob digest vs. segment id; cross-file caching in a session.
- **Async vs. blocking:** running `wasmtime` on the IO runtime's blocking pool vs. a dedicated
  decode pool.

[`WasmEncodingPlugin`]: ../../vortex-wasm/src/plugin.rs
[`register_wasm_encodings`]: ../../vortex-wasm/src/plugin.rs
[`WasmKernel`]: ../../vortex-wasm/src/kernel.rs
[`WasmKernelLoader`]: ../../vortex-wasm/src/loader.rs
[`WasmEncoding`]: ../../vortex-wasm-guest/src/encoding.rs
[`fastlanes`]: https://crates.io/crates/fastlanes
[`fsst`]: https://crates.io/crates/fsst-rs
[`onpair`]: https://crates.io/crates/onpair
