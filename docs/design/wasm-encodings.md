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

### File format: kernels as postscript-referenced segments (next)

Kernels are written as ordinary segments near the end of the file and referenced from the
**postscript** with their encoding ids:

```
table Postscript {
    dtype: PostscriptSegment;
    layout: PostscriptSegment;
    statistics: PostscriptSegment;
    footer: PostscriptSegment;
    // NEW: embedded decoder kernels, one per encoding id.
    wasm_kernels: [WasmKernelSpec];
}

table WasmKernelSpec {
    /// The array encoding id this kernel decodes (e.g. "fastlanes.bitpacked").
    id: string (required);
    /// ABI version the kernel was built against.
    abi_version: uint32;
    /// Location of the `.wasm` blob.
    segment: PostscriptSegment;
}
```

The writer gains an option to attach kernels (e.g. `with_wasm_kernel(id, bytes)`); at open, the
reader fetches the referenced segments and calls [`register_wasm_encodings`] on the scan session
before building contexts. Kernels are naturally content-addressable for dedup across files.

*Status: the session-level machinery (registration, plugin, kernel, ABI, kernels, parity tests)
is implemented; the postscript flatbuffer field and the writer/reader plumbing are the remaining
step (blocked in the current environment on regenerating flatbuffers — `flatc` — and the
workspace `Cargo.lock`).*

## Crates

- **`vortex-wasm` (the host)** — depends on `vortex-array`, `vortex-session`, `arrow-*`, and
  `wasmtime`. Provides [`WasmKernel`]/`WasmDecoder` (the runtime), `convert` (the boundary),
  [`WasmEncodingPlugin`] (the `ArrayPlugin` adapter), and [`register_wasm_encodings`].
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

Hence two result shapes, and a per-child access mode:

| | value-producing | re-arranging |
| --- | --- | --- |
| child access | `ChildMode::Values` — canonicalized and copied into the sandbox | `ChildMode::Reference` — resolved lazily, in its own encoding, never copied |
| result | `Decoded::Primitive` / `Bool` / `VarBinView` | `Decoded::Take { values_slot, indices }` |
| guest bytes moved | O(len) | 0 for the referenced child |

`Take` is deliberately the *only* re-arrangement op today. The full vocabulary a complete design
needs (`Patch`, `Mask`, `Concat`, `Slice`, `Constant`, …) is sketched in
[Toward a plan vocabulary](#toward-a-plan-vocabulary), but each of those is blocked on host work,
so shipping them speculatively would add untested attack surface. `Take` alone is enough to prove
the model on run-end, and it composes forward: the eventual arena of plan nodes has `Take` as one
node type.

## The encoding trait (guest)

A kernel is the portable mirror of a native `VTable::deserialize`. Because only the encoding
knows its children's dtypes, decoding is two-phase:

```rust
pub trait WasmEncoding {
    /// From the metadata (and the serialized child count), declare each child's dtype, length,
    /// and access mode.
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>>;

    /// Produce the node's output: either materialized bytes, or a gather over a referenced child.
    fn decode(node: &NodeView<'_>) -> GuestResult<Decoded>;
}

export_wasm_encoding!(MyEncoding); // defines vx_alloc + vx_children + vx_decode
```

`ChildSpec::values(dtype, len)` declares a child the guest will read; `ChildSpec::reference(...)`
declares one it will only *name*. Reference children are never canonicalized and never enter the
sandbox, so **their dtype is unconstrained** — this is what lets one dtype-agnostic kernel cover
cases the guest has no code for. `ChildDType` is `Parent` (the node's own dtype), primitive, bool,
or utf8; `Parent` is all a re-arranging kernel needs, since it is naming rather than reading.

`NodeView` exposes the metadata bytes (parse with `proto`), the raw buffers (resident in guest
memory, 8-byte aligned so they can be cast in place), and typed views of the `Values` children.
`Decoded` is a materialized primitive/utf8 output, or `Decoded::Take`.

## Host / guest ABI (`abi_version = 2`)

All integers little-endian; the single linear memory is exported as `"memory"`. The ABI is
**push-based**: there are no host callbacks during decode.

Guest exports:

- `vx_alloc(len) -> ptr` — bump allocation; the host uses it to place all inputs.
- `vx_children(frame_ptr, frame_len) -> ptr` — input
  `[u64 len][u32 flags][u32 n_children][u32 metadata_len][metadata]`; output `[u32 n]` + `n`
  16-byte descriptors `[u8 tag][u8 ptype][u8 nullable][pad][u64 len]`.
- `vx_decode(frame_ptr, frame_len) -> ptr` — input
  `[u64 len][u32 flags][u32 metadata_len][u32 n_buffers][u32 n_children][metadata]`
  `[(ptr,len) x buffers][(array_ptr,schema_ptr) x children]`; output points at the decoded
  array's buffer-table descriptor (or, for a gather, the child slot plus the index array).

The `flags` word carries the parent dtype: bit 0 nullability, bits 8-15 the kind
(primitive/bool/utf8), bits 16-23 the ptype. Negative returns are error codes; panics become
traps, which the host surfaces as decode errors.

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
chunk and a partial trailer. Patches overwrite `index - patches.offset`; the validity child
carries through. Scope: 4-byte primitives — other widths are pure monomorphization at ~25 KB of
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
honoured — and returns `Decoded::Take`. The host resolves the values child in its own encoding and
calls `ArrayRef::take`, which builds a lazy `DictArray`: no canonicalization, no copy, no
materialized output.

The consequences are worth stating plainly, because they are the argument for the whole design:

- **The kernel is dtype-agnostic.** The *native* decoder needs three separate implementations
  (bool / primitive / varbinview) and `vortex_bail!`s on anything else. This kernel has none —
  run-end over strings works with zero string code in the guest, and is covered by a test.
- **It is 5.3 KB**, versus 51 KB for bitpacked. Not decoding is cheap.
- **Validity falls out.** Run-end's output validity is the values' validity gathered through the
  same runs; `take` reproduces that, so the kernel never touches validity.

## Toward a plan vocabulary

`Take` is the first of what should become a small arena of dtype-agnostic ops — `Patch`, `Mask`,
`Concat`, `Slice`, `Constant`, `Struct`/`List` construction — with materialized buffers as leaves,
so hybrids compose (bitpacked would become `Patch{materialized, indices_child, values_child}`,
which also removes its current 4-byte patch-value limit). An adversarial review of the full
vocabulary returned *needs-rework*, and those findings gate the expansion:

- The arithmetic ops `for`/`bytebool`/`datetimeparts` would need (`WrappingAdd`, `Shl`, `Or`)
  **do not exist in vortex-array**; the nearest are checked/saturating, which would be a
  correctness regression. They must land natively first.
- **Statistics must never be load-bearing for safety** — a malicious file can write a small
  `Stat::Max`. Index bounds are therefore recomputed from the materialized buffer (see
  `validate_indices`), never read from stats. This rule is already followed here.
- Variadic nodes (`Concat`, `Struct`) break "acyclic by construction" if operands are an implicit
  arena range, and blow a fixed node cap.
- Child resolution must be **memoized**: `SerializedArrayChildren::get` re-decodes the subtree on
  every call, so a plan naming one slot 250 times would decode it 250 times.
- Every interior node's length must be bounded by the node length, or a small file can drive an
  arbitrarily large host allocation.
- The parent-dtype channel must become a real descriptor table (level-order, not preorder with a
  `child0` index, which is wrong for nested dtypes) before `struct`/`list` kernels are possible.

## Other ABI gaps found by the survey (not yet fixed)

- **The parent dtype is 24 bits of frame flags.** Anything that is not primitive/bool/utf8 arrives
  as `Other`. This is *fatal, not slow*, for `datetimeparts` (needs the Timestamp `TimeUnit` and
  tz), `decimal`/`decimal_byte_parts` (precision/scale), and `fixed_size_list` (`list_size`).
- **A kernel cannot *read* a string child.** Only primitive and boolean children are deliverable
  into the sandbox. `ChildMode::Reference` covers every re-arranging encoding (the child never
  enters the guest), so this only binds a hypothetical kernel that must inspect string bytes.
- **Decimal output** (i128/i256) and its 16/32-byte buffer alignment are still unsupported.
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
   lets a kernel name a child without the host canonicalizing or copying it, and `Decoded::Take`
   lets the host perform the gather with `ArrayRef::take` (a lazy `DictArray`). Proven by the
   `vortex.runend` kernel — 7.7 KB, dtype-agnostic, and correct over strings, which the native
   decoder needs a dedicated implementation for. Untrusted gather indices are validated by
   recomputing bounds, never from statistics.
5. **Untrusted-input hardening (done):** `SerializedArray::decode` and the `ArrayChildren` blanket
   impl now return `VortexError` instead of `assert!`-ing, so a lying kernel cannot abort the
   process.
6. **File plumbing (next):** the postscript `wasm_kernels` field, a writer option to attach
   kernels, and reader-side registration at file-open (see above). Requires flatbuffer
   regeneration.
7. **Breadth (later):** the rest of the plan vocabulary (gated on the review findings above), the
   full parent-dtype channel, a Vortex-shaped output vocabulary, more kernels (dict, ALP, ...),
   kernel dedup + cross-file caching, CPU-time limits, and the `wasm32` fallback runtime for the
   browser reader.

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
[`WasmEncoding`]: ../../vortex-wasm-guest/src/encoding.rs
[`fastlanes`]: https://crates.io/crates/fastlanes
[`fsst`]: https://crates.io/crates/fsst-rs
