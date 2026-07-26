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
| `encodings/runend/wasm` | `vortex.runend` | nothing — it gathers instead of decoding (5.3 KB) |
| `encodings/fsst/wasm` | `vortex.fsst` | [`fsst`] (fsst-rs)'s `Decompressor` |

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
  decoded by each other's code. (`Registry::clone` shares its map; `Registry::fork` is the
  independent copy this needs.)

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

## Two kinds of encoding, two result shapes

The axis that matters is **not** leaf-vs-nested, and it is **not** buffers-vs-children (5 of the 6
"leaf" encodings have zero buffers and read a child). It is:

> Does the decode produce **new element values**, or is the output a **permutation, subset, or
> overlay** of a child's existing values?

- **Value-producing** (bit-packing, FSST, zstd, delta, zigzag, ALP): the output bytes exist
  nowhere until the kernel computes them. Delta is the extreme case — Vortex has no scan kernel,
  so it cannot be delegated even in principle. These kernels must **materialize**.
- **Re-arranging** (run-end, dict, sparse, patched, chunked, masked, struct, list, extension):
  the output is the child, reordered/overlaid. Making the guest materialize these is wrong four
  ways: it forces the guest to reproduce dtypes it cannot even name, doubles the boundary
  crossings, blows up memory (chunked would canonicalize every chunk into a 4 GiB sandbox), and
  destroys host pushdown. These kernels must **not** materialize.

Roughly 20 of ~29 surveyed encodings are re-arranging. So the ABI's job is to let a kernel say
*"my output is child N, gathered like this"* without ever touching child N.

Hence a per-child access mode, and a result that is a **plan** rather than an array:

| | value-producing | re-arranging |
| --- | --- | --- |
| child access | `ChildMode::Values` — canonicalized and copied into the sandbox | `ChildMode::Reference` — resolved lazily, in its own encoding, never copied |
| result | a plan ending in one `Materialized` node | a plan of `Child` / `Take` / `Slice` / `Concat` / `Constant` / `SetValidity` nodes |
| guest bytes moved | O(len) | 0 for the referenced child |

See [The plan vocabulary](#the-plan-vocabulary).

## The encoding trait (guest)

A kernel is the portable mirror of a native `VTable::deserialize`. Because only the encoding
knows its children's dtypes, decoding is two-phase:

```rust
pub trait WasmEncoding {
    /// From the metadata (and the serialized child count), declare each child's dtype, length,
    /// and access mode.
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>>;

    /// Describe the node's output as a plan over its children, returning the root node.
    fn decode(node: &NodeView<'_>, plan: &mut PlanBuilder) -> GuestResult<NodeId>;
}

export_wasm_encoding!(MyEncoding); // defines vx_alloc + vx_children + vx_decode
```

`ChildSpec::values(dtype, len)` declares a child the guest will read; `ChildSpec::reference(...)`
declares one it will only *name*. Reference children are never canonicalized and never enter the
sandbox, so **their dtype is unconstrained** — this is what lets one dtype-agnostic kernel cover
cases the guest has no code for. The dtype is a [`DTypeExpr`](#the-dtype-channel): a literal, or a
derivation such as `DTypeExpr::parent()`, which is all a re-arranging kernel needs since it is
naming rather than reading.

`NodeView` exposes the node's full dtype, the metadata bytes (parse with `proto`), the raw buffers
(resident in guest memory, 8-byte aligned so they can be cast in place), and typed views of the
`Values` children. `PlanBuilder` hands back a `NodeId` per node, which is the only way to name
one — so a kernel cannot express a dangling or cyclic reference even by accident.

## Host / guest ABI (`abi_version = 3`)

All integers little-endian; the single linear memory is exported as `"memory"`. The ABI is
**push-based**: there are no host callbacks during decode.

Guest exports:

- `vx_abi_version() -> u32` — the ABI the kernel was built against. The host reads it at compile
  time and refuses a kernel that disagrees with its own, so a stale kernel fails loudly instead of
  misreading frames.
- `vx_alloc(len) -> ptr` — bump allocation; the host uses it to place all inputs.
- `vx_children(frame_ptr, frame_len) -> ptr` — input
  `[u64 len][u32 flags][u32 n_children][u32 dtype_len][u32 metadata_len][dtype][metadata]`; output
  `[u32 n]` + `n` descriptors `[u8 mode][pad x3][u32 dtype_len][u64 len][dtype]`.
- `vx_decode(frame_ptr, frame_len) -> ptr` — input
  `[u64 len][u32 flags][u32 dtype_len][u32 metadata_len][u32 n_buffers][u32 n_children]`
  `[dtype][metadata][(ptr,len) x buffers][child_entry x children]`; output points at a
  [plan](#the-plan-vocabulary) frame.

`flags` bit 0 is the parent's nullability, kept because it is free and it is what most kernels
branch on; the full type travels as a real [dtype expression](#the-dtype-channel). Negative
returns are error codes; panics become traps, which the host surfaces as decode errors.

### Array boundary: Vortex's own layouts, not Arrow

Arrays cross in Vortex's canonical layouts — a **buffer table plus a shape tag** — with no schema
and no Arrow dependency.

The boundary *was* the Arrow C Data Interface. It was removed because Arrow C FFI is a
**schema-carrying protocol and this boundary has no schema to carry**: the host already holds the
node's `DType`, and the guest declares its children's dtypes itself. The round trip therefore had
the guest write a format string that the host parsed back into a type it already knew, ran
`ArrayData::try_new` revalidation over, and then converted into Vortex's representation. For
strings that conversion was also lossy in cost: Arrow utf8's i32 offsets import as `VarBin`, which
is **not** canonical, so every string kernel paid a second full conversion of the heap.

- `shape` is `Primitive` (one values buffer), `Bool` (one bitmap), or `VarBinView` (16-byte views
  plus the data buffers they reference — Vortex's canonical string form, which FSST now emits
  directly).
- `validity` is an **algebra** — `NonNullable | AllValid | AllInvalid | Bitmap` — so a
  non-nullable or all-valid array transmits no bitmap at all. The Arrow-shaped channel copied one
  for nothing.
- Only primitive and boolean children are deliverable *into* the guest. That is not a limitation
  in practice: anything else is declared `Reference` and never enters the sandbox.
- Bitmaps are byte-aligned via `shrink_offset` before crossing, closing the bit-offset hazard a
  sliced array's mask would otherwise cause.

For primitives and bools these bytes are identical to Arrow's; only the schema went away. What
went with it: the `arrow-array`/`arrow-buffer`/`arrow-data`/`arrow-schema` dependencies, ~800
lines of schema recursion, metadata-blob parsing, and dictionary handling — and the attack surface
they carried.

### Memory

Kernels are `#![no_std]`; the SDK provides a grow-only bump `#[global_allocator]` over linear
memory and a trap-on-panic handler behind the default `runtime` feature. A kernel instance
decodes exactly once and its whole memory is reclaimed when the host drops the per-decode store,
so `dealloc` is a no-op and there is no free in the ABI.

**`vx_alloc` returns 8-byte-aligned offsets — this is part of the ABI.** Every host upload (the
frames, the raw buffers, the child structs) lands aligned, so kernels view typed data **in
place**: wasm32 is little-endian, matching the serialized format, so e.g. the bitpacked kernel
casts its packed buffer to `&[u32]` (`align_to`, checked) instead of copying words out.

Today the two shipped kernels disable the `runtime` feature because a dependency links `std`
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
chunk and a partial trailer. Patches overwrite `index - patches.offset` in the sandbox rather
than going through `PlanBuilder::patch`: bit-packing's patch values are always the parent's own
primitive type and there are few of them, so reading them costs less than the output-length index
array a plan-level patch needs. The validity child carries through. Scope: 4-byte primitives — other widths are pure monomorphization at ~25 KB of
unrolled unpack code per width family. Blob: **~51 KB** (~35 KB once fastlanes-rs's
`num-traits` edge stops linking `std`; the unpack kernels dominate the rest).

### `vortex.fsst` (`encodings/fsst/wasm`)

Parses `FSSTMetadata`; declares `[uncompressed_lengths, codes_offsets, validity]` children;
rebuilds the symbol table with `fsst::Symbol::from_slice` and bulk-decompresses the whole codes
heap with **the same [`fsst`] crate `Decompressor` the native canonical path uses**; the prefix
sums of the uncompressed lengths are exactly the output utf8 offsets. Blob: **~26 KB**.

### `vortex.runend` (`encodings/runend/wasm`) — the structural case

Run-end is the canonical re-arranging encoding, and its kernel decodes **nothing**. It declares
`ends` as `Values` (to build indices) and `values` as `Reference`, expands the run ends into one
`u32` gather index per row — mirroring `trimmed_ends_iter`, so a sliced array's `offset` is
honoured — and returns the plan `take(child(VALUES), indices)`. The host resolves the values child
in its own encoding and calls `ArrayRef::take`, which builds a lazy `DictArray`: no
canonicalization, no copy, no materialized output.

The consequences are worth stating plainly, because they are the argument for the whole design:

- **The kernel is dtype-agnostic.** The *native* decoder needs three separate implementations
  (bool / primitive / varbinview) and `vortex_bail!`s on anything else. This kernel has none —
  run-end over strings works with zero string code in the guest, and is covered by a test.
- **It is 5.3 KB**, versus 51 KB for bitpacked. Not decoding is cheap.
- **Validity falls out.** Run-end's output validity is the values' validity gathered through the
  same runs; `take` reproduces that, so the kernel never touches validity.

## The plan vocabulary

A kernel returns a **plan**, not an array: a small tree of operations over the node's children
that the host evaluates. The design constraint that shapes everything else is that a kernel exists
*precisely because the reader lacks that encoding*, so a plan may only name operations the reader
is guaranteed to have. Every opcode is therefore one `vortex-array` constructor:

| op | operands | evaluated as | needed by |
| --- | --- | --- | --- |
| `Materialized` | dtype + array descriptor | `convert::ArrayDescriptor::build` | bit-packing, FSST, zstd, ALP — anything computing new values |
| `Child` | slot | the serialized child, in its own encoding | every re-arranging encoding |
| `Take` | base, indices | `ArrayRef::take` → `DictArray` | run-end, dict |
| `Slice` | base, start, stop | `ArrayRef::slice` → `SliceArray` | windowed children |
| `Concat` | parts… | `ChunkedArray::try_new` | chunked; and patching, below |
| `Constant` | scalar, len | `ConstantArray::new` | sparse's fill value |
| `SetValidity` | base, mask | `MaskedArray::try_new` | encodings storing validity as a separate child |

Two ops that a first sketch had, and that turned out not to be needed:

- **`Patch`** is not a primitive. `patch(base, indices, values)` is exactly
  `take(concat[base, values], merged)`, where `merged[i]` is either `i` or `base.len() + j`. The
  guest SDK offers `PlanBuilder::patch` as sugar that emits those three nodes, so kernel authors
  get the ergonomic op while the host's trusted evaluator stays smaller by one case. `PatchedArray`
  would not have served anyway: it is a lane-transposed FastLanes-specific structure with
  chunk-local `u16` indices, not a general overlay.
- **`RunEnd`/`Dict`** are not ops. They *are* the encodings a kernel is standing in for, so
  requiring them of the host would defeat the purpose.

### Why the wire format is flat

The plan is a flat postorder array, not a nested tree:

```text
[u32 n_nodes][u32 root][u32 aux_len][u32 reserved]
[node × n_nodes]        node = [u8 op][u8 flags][u16 pad][u32 a][u32 b][u32 c]
[aux bytes]
```

A node may only reference nodes at a **lower index**. This is the whole safety argument, and it
buys three things at once:

1. **Cycles are unrepresentable**, not merely rejected.
2. **Evaluation is a `for` loop** over a slot table — no recursion, so no depth to bound and no
   host stack to overflow. A nested encoding would have handed an untrusted file a recursive
   descent parser.
3. **Sharing is free.** References are by index, so a node used twice is evaluated once; the plan
   is a DAG.

Payloads that do not fit in three `u32`s — 64-bit ranges, scalars, `Concat`'s operand list, a
`Materialized` descriptor — live in a trailing `aux` blob.

### What the host validates

The plan is untrusted file data, so every one of these is checked, and none of them may rely on
anything the file says about itself:

| checked at | check |
| --- | --- |
| parse | node count ≤ 1024; root in range; opcode known; **every operand strictly backwards**; aux payloads present and in range |
| `Take` | indices are non-nullable unsigned primitives, and **every index is recomputed** against the base's length |
| `Slice` | `start ≤ stop ≤ base.len()` |
| `Concat` | ≤ 1024 parts, all of one dtype |
| `SetValidity` | mask is a non-nullable boolean of the base's length |
| `Constant` | scalar dtype is null/bool/primitive; length within budget |
| every node | running total of intermediate rows ≤ 64× the node's length |
| root | length and dtype match the node being decoded |

Two of these deserve their reasoning spelled out.

**Index bounds are recomputed, never read from statistics.** `Stat::Max` is itself
attacker-controlled file data. It matters here because `ArrayRef::take` builds a `DictArray`,
whose constructor checks only that codes are integral — an out-of-range index would surface later
as a panic or as data from beyond the child. (The same reasoning applies to `Patches::new`, which
in release builds only *debug*-asserts sortedness and bounds-checks the last index; that is why
patching goes through `take`, which validates every index, rather than through `Patches`.)

**A plan is cheap to write and can describe expensive work.** Twelve `Concat` nodes describe an
array 4096× the size of their input. The ops build lazy arrays, so nothing blows up during
evaluation — but the cost is real once the scan canonicalizes, so the running output total is
capped at a multiple of the node's own length.

## The dtype channel

ABI v2 gave the guest three bits of parent dtype: primitive, bool, utf8, or `Other`. Anything else
was fatal, not slow — `datetimeparts` needs a Timestamp's `TimeUnit` and timezone, `decimal` needs
precision and scale, `fixed_size_list` needs its size, and none of them could see any of it.

v3 sends a real type, in a compact preorder encoding with a tag byte per node
(`vortex-wasm-guest/src/dtype.rs` holds the grammar). It covers every `DType` variant.

The part that makes it *complete* rather than merely wide is **derivations**. A literal is not
always writable: extension types resolve through a host vtable registry, so no byte encoding lets
a guest construct one. And a kernel generic over its parent does not *want* to name a concrete
type. So a guest may instead write a path:

```text
Parent | Field(i, inner) | Element(inner) | Storage(inner) | Nullable(inner) | NonNullable(inner)
```

These compose — `NonNullable(Element(Field(1, Parent)))` is valid — and the host resolves them
against a `DType` it already trusts. The kernel never holds the type, only a route to it. That is
how the run-end kernel stays dtype-agnostic: it declares its values child as `Parent` and works
over strings, decimals, structs, or any type added later, with no code.

The direction is asymmetric on purpose: the host only ever writes literals, because it holds the
real type and has nothing to derive from.

## Remaining ABI gaps

- **A kernel cannot *read* a string child.** Only primitive and boolean children are deliverable
  into the sandbox. `ChildMode::Reference` covers every re-arranging encoding (the child never
  enters the guest), so this only binds a hypothetical kernel that must inspect string bytes.
- **Decimal output** (i128/i256) and its 16/32-byte buffer alignment are still unsupported.
- **Arithmetic ops are still missing natively.** `for`/`bytebool`/`datetimeparts` would want
  `WrappingAdd`, `Shl`, `Or` as plan nodes; `vortex-array`'s nearest equivalents are
  checked/saturating, which would be a correctness regression. They must land natively before the
  vocabulary can grow to cover those encodings.
- **Child resolution is not memoized.** `SerializedArrayChildren::get` re-decodes the subtree on
  every call, so a plan naming one slot 250 times decodes it 250 times. The node cap bounds this,
  but a memo in the resolver closure is the real fix.
- **`vortex.chunked` is inexpressible at any cost**: its per-chunk lengths are the *decoded
  contents of child 0*, and `children` is a single pure call with a mandatory length. Fixing it
  requires an iterative declaration phase, not more plan ops.

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
child/buffer counts and schema recursion depth, validate every guest-returned structure through
Arrow's own `ArrayData::try_new`, and treat any guest trap as a decode error (never a host
panic). CPU-time bounding (wasmtime fuel or epoch interruption) is a planned follow-up. The
kernel is untrusted data from the file, exactly like array bytes; a buggy kernel can only corrupt
*that array's* values, never host memory.

## Binary size

Compiled `wasm32-unknown-unknown`, size-optimized (`opt-level = "z"`, `lto`, `panic = "abort"`,
`strip`):

| kernel | size | notes |
|---|---|---|
| minimal SDK kernel (no_std, no deps) | ~4 KB | the SDK floor: allocator + buffer glue |
| `vortex.fsst` | ~26 KB | fsst-rs `Decompressor` + `std` (fsst-rs is not yet no_std) |
| `fastlanes.bitpacked` | ~51 KB | fastlanes unrolled unpack kernels + `std` via num-traits |
| `fastlanes.bitpacked`, fully no_std | **~35 KB** | measured with num-traits `default-features = false` patched into fastlanes-rs |

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
   [the array boundary](#array-boundary-vortexs-own-layouts-not-arrow).
3. **Session-level wasm encodings (done):** `WasmLayout` removed. Kernels decode the **real
   serialized parts** (ABI v2: `vx_children` + pushed `vx_decode` frame);
   [`WasmEncodingPlugin`] registers under the encoding's id and returns decoded arrays;
   [`register_wasm_encodings`] merges kernels into a session with native-supersedes semantics.
   Kernels live alongside their encodings (`encodings/fastlanes/wasm`, `encodings/fsst/wasm`)
   and reuse the same decode crates; parity is tested against natively-serialized bytes,
   including patches and nullable columns.
4. **Structural decoding via `Take` (done):** the survey found that ~20 of ~29 encodings only
   re-arrange a child, so the guest must not materialize them. `ChildMode::{Values, Reference}`
   lets a kernel name a child without the host canonicalizing or copying it, and a gather lets the
   host perform the re-arrangement with `ArrayRef::take` (a lazy `DictArray`). Proven by the
   `vortex.runend` kernel — dtype-agnostic, and correct over strings, which the native decoder
   needs a dedicated implementation for. Untrusted gather indices are validated by recomputing
   bounds, never from statistics.
5. **Untrusted-input hardening (done):** `SerializedArray::decode` and the `ArrayChildren` blanket
   impl now return `VortexError` instead of `assert!`-ing, so a lying kernel cannot abort the
   process.
6. **File plumbing (done):** the postscript `wasm_kernels` field, `with_wasm_kernel` on the
   writer, and loader-based registration at file-open — opt-in, file-scoped, and fetching only the
   kernels the reader actually lacks (see above). A `vx_abi_version` guest export makes a stale
   kernel a clear error rather than a misread frame.
7. **Plan vocabulary and dtype channel (done, ABI v3):** `vx_decode` now returns a flat postorder
   [plan](#the-plan-vocabulary) — `Materialized`, `Child`, `Take`, `Slice`, `Concat`, `Constant`,
   `SetValidity` — each opcode one `vortex-array` constructor, evaluated in a single non-recursive
   forward pass with per-node validation and an output budget. Patching is guest-side sugar over
   `take`+`concat` rather than a host primitive. The three-bit parent-kind tag became the full
   [dtype channel](#the-dtype-channel), with derivations so a kernel can name types it cannot
   construct.
8. **Breadth (next):** more kernels (dict, ALP, sparse), the missing native arithmetic ops,
   memoized child resolution, decimal output, kernel dedup + cross-file caching, CPU-time limits,
   and the `wasm32` fallback runtime for the browser reader.

Pushdown (filter/pruning into the kernel) is explicitly **out of scope** — WASM encodings only
decompress; the engine filters on the decoded output.

## Open questions

- **Laziness:** the plugin decodes eagerly at deserialize time. A lazy wrapper array (decode on
  first execute) would let filters skip decodes for pruned ranges.
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
