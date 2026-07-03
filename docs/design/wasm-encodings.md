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
sandboxed WebAssembly module. A reader that understands the `WasmLayout` can then decode arrays
written with an encoding it has never seen, by running the embedded WASM kernel against the
serialized array and the host's existing decode machinery.

This document describes the on-disk format, the host/guest ABI, and the crate layout.

## Overview

```
┌──────────────────────── Vortex file ─────────────────────────┐
│ MAGIC                                                          │
│ … data segments (child layout, written normally) …            │
│ … WASM kernel segment (one .wasm blob, written at EOF) …      │   ← split off to end
│ DType / Layout / Statistics / Footer flatbuffers              │
│ Postscript + EOF                                              │
└───────────────────────────────────────────────────────────────┘
```

### Key principle: the data is a *normal* serialized array; the WASM layout adds only the decoder

> **Status:** this is the *target* on-disk model (phase 5 below). The current implementation uses a
> transitional payload+child write model (see [Write side](#write-side-wasmencoder)): the encoder
> produces an explicit payload plus a child array rather than the guest parsing the serialized array
> flatbuffer itself. The host/guest *boundary* described below (Arrow C Data Interface) is already
> in place; the on-disk "embed only the decoder" change is the remaining work.

An encoding that wants a WASM decoder is still implemented as an ordinary Vortex array encoding
whose data is written in the **existing serialized array format** (the `ArrayNode` flatbuffer plus
its buffers and child nodes). A `WasmLayout` wraps that standard data and additionally **embeds the
`.wasm` decoder blob — and nothing else bespoke**. The consequence:

- A reader that **has the native VTable** for the encoding decodes the bytes directly, the normal
  way, and **ignores the blob**.
- A reader that **lacks the VTable** runs the embedded WASM decoder over the **same bytes**.

So the blob is a portable fallback decoder for an otherwise-normal encoded array — never a separate
on-disk representation.

`WasmLayout` therefore holds:

1. the **child layout** holding the encoded array in the standard serialized format; and
2. a **segment id** for the embedded `.wasm` decoder (written at end-of-file).

At read time, when the native VTable is absent, the reader:

1. loads + instantiates the kernel in an embedded WASM VM (`wasmtime`), caching the compiled module;
2. hands the guest the **serialized array** (flatbuffer header + buffers) for the node to decode;
3. the guest **parses the array flatbuffer header itself** with `vortex-flatbuffers` compiled into
   the module (*without the rest of Vortex*), reading its own encoding metadata and buffers;
4. whenever the guest needs a decoded child array it calls the host import `vx_decode_child`; the
   host decodes that child node through the `VortexSession` (native encodings) and hands it back as
   **Arrow C Data Interface**;
5. the guest produces its decoded output, also as **Arrow C Data Interface**;
6. the host reads those C structs out of guest memory, deep-copies the buffers, and rebuilds a
   Vortex array via `ArrayRef::from_arrow`, yielding a Vortex array.

**Boundary formats.** *Decoded* arrays crossing the boundary in either direction (child results in,
final result out) use the **Arrow C Data Interface**. Both sides build and read those C structs
**directly as plain bytes** (no Arrow library, no nanoarrow) — the layout is fixed and documented in
[`crate::abi`](../../vortex-wasm-guest/src/abi.rs). The host cannot hand the guest a borrowed
`FFI_ArrowArray` because the boundary is wasm32 (4-byte pointers, separate address space), so it
copies buffers out of guest memory and reconstructs `arrow_data::ArrayData` itself rather than using
`arrow`'s same-address-space `from_ffi`. There is no bespoke wire format — `CanonicalMessage` is
removed.

## Crates

Two new crates, kept out of the core dependency graph so that `wasmtime` never leaks into
`vortex-array`/`vortex-layout`:

### `vortex-wasm-guest` (the guest SDK)

A small crate that an encoding author links against when building their decoder to
`wasm32-unknown-unknown`. It is **`#![no_std]` and dependency-free** — `core`/`alloc` only, never
any Vortex crate (not even `vortex-error`) and no Arrow library — which is what keeps a compiled
kernel to ~4 KB (see [Binary size](#binary-size)). It provides:

- the host/guest ABI (exported entrypoints, imported host functions) and the Arrow C struct field
  offsets, in `abi`;
- the guest **runtime** (behind the default `runtime` feature): a grow-only bump
  `#[global_allocator]` over linear memory and a trap-on-panic `#[panic_handler]`, replacing
  `std`'s dlmalloc and panic machinery — the bulk of a `std` kernel's size;
- the **allocator API**: `host::alloc`/`host::alloc_bytes` for kernel scratch/output buffers, and
  the `vx_alloc` export the host calls to place data into guest memory;
- `arrow`: build a decoded primitive output (`Decoded` → Arrow C structs) and read a host-supplied
  child (`ChildView`) — all as plain byte layout, no Arrow library;
- `host::decode_child`, the safe wrapper over the `vx_decode_child` host import;
- a minimal, formatting-free `GuestError` (a `&'static str`, no `format!`);
- a `WasmEncoding` trait plus an `export_wasm_encoding!` macro that wires up the `vx_alloc` and
  `vx_decode` exports around a user-supplied `decode` function (see
  [The encoding trait](#the-encoding-trait)).

### `vortex-wasm` (the host)

Depends on `vortex-layout`, `vortex-array`, `vortex-session`, and `wasmtime`. It provides:

- `WasmKernel` — an instantiated, reusable wrapper around a `wasmtime::Module` and the host import
  table, exposing `decode(input, &dyn HostDecoder, &VortexSession) -> ArrayRef`;
- `arrow_ffi` — the host side of the Arrow C Data Interface boundary: `import` rebuilds a Vortex
  array from C structs in guest memory; `export` writes a canonical array as C structs into guest
  memory for `vx_decode_child`;
- `WasmLayout` + `WasmLayoutEncoding` + `WasmLayoutMetadata`, the layout `VTable`
  implementation;
- `WasmReader`, the `LayoutReader` that drives the kernel;
- `WasmLayoutStrategy`, the writer that wraps a child strategy and appends the kernel segment at
  EOF;
- `register_wasm_layout(session)`, registering the encoding so files can be read.

## On-disk layout (`WasmLayout`)

`encoding = "vortex.wasm"`. Stored in the layout flatbuffer like any other layout:

| field      | meaning                                                                 |
|------------|------------------------------------------------------------------------|
| `row_count`| rows produced by the decoded output                                    |
| `metadata` | prost `WasmLayoutMetadata` (see below)                                  |
| `children` | `[data_layout]` — the encoded child input(s), each in the output dtype; empty when the encoded form lives entirely in the payload |
| `segments` | `[kernel_segment_id]` (+ optional payload segment) — the embedded `.wasm` blob |

```protobuf
message WasmLayoutMetadata {
  string  encoding_id = 1;   // guest encoding id, e.g. "acme.delta"
  uint32  abi_version = 2;   // host/guest ABI version
  bool    has_payload = 3;   // whether a payload segment follows the kernel segment
}
```

The metadata is deliberately minimal. A child layout already records its **own encoding id** in the
layout flatbuffer, and a child's **dtype is the layout's output dtype** (the dtype a native VTable
would read the same bytes with), which is itself in the file — so neither is duplicated here. The
layout flatbuffer has no per-layout dtype field; the parent supplies a child's dtype on
deserialization, and `WasmLayout` supplies its own.

The kernel itself is content-addressed: identical kernels across many `WasmLayout` nodes in one
file should share a single segment (a writer-side dedup keyed on the blob digest). For the first
cut each `WasmLayout` references one kernel segment; dedup is a follow-up.

### Writing the kernel at EOF with `SequencePointer::split_off`

`LayoutStrategy::write_stream` receives an `eof: SequencePointer` guaranteed to sort after every
chunk in the stream. To force the kernel to the very end of the file, the strategy takes a
sequence id from `eof` (via `split_off`) and uses it for the kernel segment write, while the
child data is written with the normal in-stream sequence ids. Because `SegmentSink::write`
calls `SequenceId::collapse().await`, the kernel's segment bytes are flushed only after all
earlier (data) segments — placing the `.wasm` blob at the end of the file. As the trait docs
require, the strategy awaits the child write and the kernel write **concurrently** to avoid the
EOF-deadlock.

## Host / guest ABI (`abi_version = 1`)

All integers little-endian. The single shared linear memory is exported by the guest as
`"memory"`.

### Memory

Kernels are **`#![no_std]`**, and the SDK provides the allocator: a grow-only **bump allocator**
over linear memory (starting at the linker-provided `__heap_base`, growing memory on demand),
installed as the `#[global_allocator]` behind the SDK's default `runtime` feature. `Vec` et al.
work normally via `alloc`. The guest exports `vx_alloc`, backed by the same allocator; the host
calls it to place inputs and host-decoded children into guest memory, and kernels use
`host::alloc`/`host::alloc_bytes` for their own scratch/output buffers.

There is deliberately **no free/dealloc in the ABI** and the bump allocator's `dealloc` is a no-op:
a kernel instance decodes exactly once, and its entire linear memory is reclaimed when the host
drops the per-decode store — so per-allocation bookkeeping would be dead weight. Dropping `std`
(dlmalloc + panic machinery) took kernels from ~16 KB to ~4 KB. Panics become wasm traps, which the
host surfaces as decode errors. A kernel that wants `std` or its own allocator disables the
`runtime` feature.

### Guest exports (host calls these)

- `vx_alloc(len: i32) -> i32`
  Allocate `len` bytes in guest linear memory and return the offset.
- `vx_decode(input_ptr: i32, input_len: i32) -> i32`
  Decode the input at `[input_ptr, input_ptr+input_len)`. Returns the offset of an
  `(array_ptr: u32, schema_ptr: u32)` pair pointing at the result's Arrow C Data Interface structs.
  A negative return value is an error code.

### Host imports (guest calls these), module `"vortex_host"`

- `vx_decode_child(node_index: i32, out_ptr: i32) -> i32`
  Ask the host to decode the child array at `node_index` (an index into the serialized array
  header's `children`, in document order). The host decodes it through the session, writes the
  result as **Arrow C Data Interface** structs into guest memory (allocating via the guest's
  `vx_alloc`), and stores the `(array_ptr: u32, schema_ptr: u32)` pair at `out_ptr`. Returns 0 on
  success, negative on error.
- `vx_host_log(ptr: i32, len: i32)` (optional, debug only)
  Log a UTF-8 string from guest memory.

### Decoded-array boundary: Arrow C Data Interface

Decoded arrays cross the boundary in both directions as the [Arrow C Data Interface]. Both sides
lay out the `ArrowSchema` and `ArrowArray` C structs as plain bytes in the wasm32 ABI (4-byte
pointers, 8-byte/8-aligned `int64`); the field offsets are constants shared by the guest's `abi`
module and the host's `arrow_ffi`. The `ArrowArray.buffers` pointer addresses a contiguous array of
4-byte pointers, one per buffer, that point into the same linear memory.

- `vx_decode`'s return value points at an 8-byte `(array_ptr: u32, schema_ptr: u32)` pair; each
  `vx_decode_child` call writes the same pair at its `out_ptr`.
- The host allocates space in guest memory for child structs/buffers via the guest's `vx_alloc`.
- Scope today: **primitive and boolean** arrays in both directions, plus **utf8 strings** for
  kernel *output* (guest write + host import; string children are not yet exported to guests). A
  primitive/bool array exposes two buffers in Arrow order — buffer 0 the validity bitmap, buffer 1
  the values; utf8 exposes three — validity, `i32` offsets (`len + 1`), and the concatenated data —
  matching Arrow's own layouts.

**Nullability.** The schema's `flags` carries `ARROW_FLAG_NULLABLE` (2). When set, buffer 0 is a
validity bitmap (`ceil(len / 8)` bytes, LSB-first, 1 = valid) and `null_count` is `-1` (unknown);
when clear the validity pointer is null. The values buffer always holds an entry at every position —
null slots may contain arbitrary bytes. The host turns a present bitmap into a `Validity::Array`.

[Arrow C Data Interface]: https://arrow.apache.org/docs/format/CDataInterface.html

## The encoding trait

The Rust surface an encoding author implements over the ABI above.

### Today (transitional payload+child model)

```rust
/// A decoder for a single WASM-embedded Vortex encoding.
pub trait WasmEncoding {
    /// Decode `input` (the encoding-specific bytes the host passes to `vx_decode`).
    fn decode(input: &[u8]) -> GuestResult<Decoded>;
}

export_wasm_encoding!(MyEncoding); // defines the `vx_alloc` + `vx_decode` exports
```

`input` is the opaque payload the write-side [`WasmEncoder`](#write-side-wasmencoder) produced;
child inputs are fetched by index with `host::decode_child(i) -> ChildView` (an Arrow view over
guest memory); the returned [`Decoded`] is laid out as Arrow C structs by the SDK. The trait is
static (no `&self`): a kernel is a decoder for exactly one encoding, and the macro monomorphises
the exports around it.

### Target (phase 5: decode the serialized array itself)

Once a `WasmLayout` embeds *only* the decoder and the data stays in the existing serialized array
format, the trait's input becomes a parsed view of the **serialized array node** rather than a
bespoke payload — the wasm mirror of a native `VTable::deserialize`:

```rust
pub trait WasmEncoding {
    /// Stable encoding id; must match `WasmLayoutMetadata.encoding_id` so the host can pair the
    /// blob with the layout (and dedup kernels).
    const ID: &'static str;

    /// Decode one serialized array node into canonical output.
    fn decode(array: &ArrayView<'_>) -> GuestResult<Decoded>;
}
```

where `ArrayView` wraps the `ArrayNode` flatbuffer the guest parses itself (generated code is pure
`flatbuffers` + `alloc`):

```rust
impl ArrayView<'_> {
    /// Logical element count of this node.
    pub fn len(&self) -> usize;
    /// This encoding's own metadata bytes from the `ArrayNode` (e.g. FoR's reference, a
    /// bit-packer's bit width) — the same bytes a native `VTable::deserialize` would read.
    pub fn metadata(&self) -> &[u8];
    /// This node's raw data buffers (e.g. the packed bitstream).
    pub fn nbuffers(&self) -> usize;
    pub fn buffer(&self, i: usize) -> &[u8];
    /// Child nodes: decoded *by the host* through `vx_decode_child` (native encodings decode
    /// natively; nested wasm encodings recurse), returned as an Arrow view.
    pub fn nchildren(&self) -> usize;
    pub fn child(&self, i: usize) -> GuestResult<ChildView>;
}
```

The split mirrors native decoding exactly: **metadata + own buffers** are interpreted by the
kernel; **children** are delegated back to the host session. `export_wasm_encoding!` stays the
same glue, with `vx_decode` receiving the serialized bytes instead of a payload.

On the host side the write-time counterpart shrinks accordingly: today's `WasmEncoder`
(payload + child) disappears, and a native encoding that wants a portable fallback just registers
its kernel —

```rust
/// Host-side: pairs a native encoding with the embedded decoder able to read its serialized form.
pub trait WasmDecodeFallback: Send + Sync {
    /// Must equal the guest kernel's `WasmEncoding::ID`.
    fn encoding_id(&self) -> &str;
    /// The compiled `.wasm` decoder blob to embed (content-addressed for dedup).
    fn kernel(&self) -> ByteBuffer;
}
```

— and `WasmLayoutStrategy` writes the already-encoded array through the normal serialized format,
attaching the kernel segment.

## Reader flow (`WasmReader`)

WASM layouts are **decode-only**: the kernel decompresses and nothing more. There is no pushdown
and no statistics-based pruning — filters and projections are evaluated on the fully decoded array,
exactly as a `Flat` leaf does. This keeps kernels simple and keeps untrusted, file-supplied code
off the query-planning path.

`WasmReader` builds one child reader per child layout (propagating `LayoutReaderContext`). Its
`projection_evaluation`:

1. fetches and compiles the kernel from its segment;
2. eagerly decodes each child input through the normal layout reader machinery into a canonical
   array — these back the `vx_decode_child` host import, which exports them as Arrow C structs on
   demand;
3. fetches the optional payload segment;
4. runs `WasmKernel::decode(payload, decoder, session)`, then slices to the row range, applies the
   input row mask, and evaluates the projection expression on the decoded array.

`filter_evaluation` is the same decode-then-evaluate path returning a refined mask;
`pruning_evaluation` returns the input mask unchanged. Neither pushes anything into the kernel.

### Why the guest parses the header

The requirement that the guest parse the array flatbuffer header *without the rest of Vortex* is
satisfied by `vortex-flatbuffers`: it only depends on `flatbuffers`, `vortex-buffer`, and
`vortex-error`, and builds for `wasm32-unknown-unknown`. The guest therefore reads `encoding`,
`metadata`, the buffer table, and `children` straight from the flatbuffer, giving the encoding
full control over how it interprets its own metadata and buffers — exactly mirroring what a
native Vortex `VTable::deserialize` would do, but sandboxed.

## Write side (`WasmEncoder`)

`WasmLayoutStrategy` pairs a kernel with a `WasmEncoder`, the write-side counterpart of the
kernel. For each input chunk the encoder returns a `WasmEncoded { payload, child }`: the `payload`
bytes the guest parses, and an **optional** child input array the kernel decodes. A child, when
present, carries the layout's output dtype; an encoding whose entire encoded form fits in the
payload returns `child: None`. The strategy writes any child through a child strategy, the payload
as its own segment, and the kernel once at end-of-file; multiple chunks are wrapped in a
`ChunkedLayout` sharing the one kernel segment. `IdentityEncoder` (empty payload, chunk as child) is
the trivial case.

## Worked example: Frame of Reference (the minimal real encoding)

FoR is the smallest encoding that actually transforms data, so it is the reference example.

- **Write** (`ForEncoder`, host): pick a reference (the column minimum), store it in the payload
  (`[i32 reference]`), and store `value - reference` as the child deltas array.
- **Read** (the FoR kernel, guest): read the reference from the payload, decode the child deltas
  via `vx_decode_child`, and emit `reference + delta[i]`.

Both halves live as runnable code:

- `vortex-wasm-guest/examples/for-kernel/` — the FoR kernel in Rust, built on the guest SDK
  (`host::decode_child`, `Decoded`, `export_wasm_encoding!`), compiling to
  `wasm32-unknown-unknown`.
- `vortex-wasm/tests/kernel_roundtrip.rs` — the host `ForEncoder` plus the **compiled** FoR kernel
  (committed under `tests/fixtures/`, `include_bytes!`-ed), writing and reading a FoR `WasmLayout`
  end to end through real layout machinery and the real guest.

## Worked example: `vortex.fastlanes.bitpacked` (native-parity semantics)

`bitpacked-kernel` decodes the **real** FastLanes bit-packed encoding — not a simplified stand-in.
The kernel links the same [`fastlanes`] crate the native `BitPacked` VTable uses, so the packed
layout (1024-element chunks of `128 * bit_width` bytes in the transposed lane order) is decoded
bit-for-bit identically, and it honours the encoding's full semantics:

- **`offset`** — a slice into the first chunk (from array slicing);
- a **partial final chunk** — unpacked into scratch and truncated;
- **patches** — values wider than `bit_width`, stored separately by the encoder and overwriting the
  unpacked output at their positions.

The round-trip test packs with the native `BitPackedData::encode` (the exact code the real encoding
runs, including automatic patch extraction for outliers) and asserts the kernel reproduces the
original values — with and without patches.

Semantic parity has a size cost: the compiled kernel is **~44 KB**, vs ~4 KB for the simple
kernels. About 28 KB is the `fastlanes` crate's unrolled per-width unpack kernels (all 33 widths
stay reachable because `bit_width` is a runtime value), and ~12 KB is `std` linked back in because
`fastlanes`' `num-traits` dependency does not set `default-features = false` — fixing that upstream
in `fastlanes-rs` would drop the kernel to ~32 KB. Kernels are read once per file and cached, so
this remains acceptable.

This also shows the two shapes an encoding can take. FoR keeps a **child** (the deltas, in the
output dtype) decoded via `vx_decode_child`; bitpacked folds its entire encoded form into the
**payload** and has no child. Because a child always carries the layout's output dtype, child dtypes
are never stored in the metadata — an encoding that needs a differently-typed buffer carries it in
the payload instead.

[`fastlanes`]: https://crates.io/crates/fastlanes

## Worked example: FSST (a real string encoding)

FSST replaces frequent 1-8 byte substrings with 1-byte codes from a table of up to 255 symbols;
code 255 escapes a literal byte. It is the ideal wasm-encoding shape: **training and compression
are complex and stay on the write side** (the `fsst` crate on the host), while **decompression is a
trivial table walk** — so the decoder that ships in the file is tiny.

- **Write** (`FsstEncoder`, host): train a `fsst::Compressor` over the chunk's strings, compress
  each, and pack everything into the payload:
  `[u32 n_symbols][symbols as u64 LE][symbol lengths][u32 n_strings][(n+1) u32 code offsets][codes]`.
  No child.
- **Read** (`examples/fsst-kernel`, guest, ~40 lines): walk the codes — escape → literal byte,
  else copy `symbol_lengths[code]` bytes of `symbols[code]` — building the Arrow utf8 offsets +
  data buffers directly. Output is a `Decoded::Utf8`.

The compiled kernel is **~4.1 KB**, and the payload lands well under the raw string bytes on
repetitive data (>2× on URL-like strings in the test). FSST is also what drove the first
non-primitive output type across the Arrow boundary (`DecodedUtf8` guest-side, `Utf8` import
host-side); Vortex's native reader canonicalizes the imported utf8 to `VarBinView` as usual.

## Binary size

The prototype example kernels (`wasm32-unknown-unknown`, size-optimized: `opt-level = "z"`, `lto`,
`codegen-units = 1`, `panic = "abort"`, `strip`) were ~69–74 KB. That is **almost entirely the
guest's `vortex` dependencies, not Rust `std`**:

| guest | size |
|---|---|
| zero-dependency (core + std + alloc only) | **~5.9 KB** |
| prototype kernel (via `vortex-error` + `vortex-flatbuffers` + `vortex-buffer`) | ~74 KB |
| dependency-free SDK, `std` (Arrow C structs + `GuestError`) | ~16 KB |
| dependency-free SDK, **`#![no_std]` + SDK bump allocator** (current) | **~3.9–4.3 KB** |
| bitpacked kernel (real `fastlanes` unpack kernels + `std` via `num-traits`) | ~44 KB |

`vortex-error` is the dominant cost: it pulls in `jiff`, `prost`, and `arrow-schema` as
non-optional dependencies, none of which a kernel needs. `vortex-flatbuffers` then drags
`vortex-error` in transitively. Dropping all vortex deps got kernels from ~74 KB to ~16 KB; the
remaining bulk was `std`'s dlmalloc allocator and panic machinery. **Kernels are now `#![no_std]`**
with the SDK's grow-only bump allocator and trap-on-panic handler (see [Memory](#memory)), taking
the simple example kernels to **3.9–4.3 KB**. Kernels that reuse real encoding libraries pay for
what they use — see the [bitpacked example](#worked-example-vortexfastlanesbitpacked-native-parity-semantics).

**The guest SDK must therefore avoid `vortex-error` entirely** and use a minimal, formatting-free
error type (a `GuestError` carrying a `&'static str`, no `format!`). Two facts make this clean:

- The **decoded-array boundary is Arrow C Data Interface**, which is pure byte layout — the guest
  builds/reads it with only `core`/`alloc`, no vortex crates.
- The **generated flatbuffer code is pure `flatbuffers` + `alloc`** (zero vortex references), so the
  guest can parse the serialized array header by depending on the `flatbuffers` crate plus the
  generated `array`/`dtype` modules, **without** `vortex-flatbuffers`'s `vortex-error`/`vortex-buffer`
  (either by depending on `vortex-flatbuffers` with its trait helpers feature-gated off, or by
  `include!`-ing the generated modules directly).

Current guest dependency set: `core`/`alloc` only (~4 KB kernels). The phase-5 target adds
`flatbuffers` for parsing the serialized array header — expected to stay in the single-digit to
low-tens of KB.

## Output format

A kernel returns its decoded array as **Arrow C Data Interface** structs (see the
[decoded-array boundary](#decoded-array-boundary-arrow-c-data-interface)). The host rebuilds a
Vortex array from them with `ArrayRef::from_arrow`, so once imported the result is in Vortex's
native representation and re-encodes to canonical encodings like any other array. Arrow was chosen
over a bespoke Vortex wire format because it needs **no Vortex dependency in the guest** — the C
struct layout is plain bytes the guest writes directly — which is the key to keeping kernels small.

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

We use the default **Cranelift** backend (not the newer Winch/Pulley, which are less battle-tested)
and compile each kernel once (`WasmKernel::new`), instantiating a fresh `Store` per decode.

> **`wasm32` / browser caveat.** Neither `wasmtime` nor `wasmer` can execute guest wasm while the
> runtime *itself* is compiled to `wasm32-unknown-unknown` (wasmtime's `runtime` feature does not
> build for wasm32; wasmer only delegates to the host's `WebAssembly` engine there). Only `wasmi`
> (a pure-Rust interpreter) self-hosts in wasm32. Vortex does target `wasm32-unknown-unknown` (the
> `wasm-test` crate, `vortex-web`), but `vortex-wasm` is not in that build today (the `vortex`
> umbrella does not depend on it), so this is not a current blocker. If the browser reader ever
> needs to decode WASM-encoded files, the clean path is to select `wasmi` behind
> `#[cfg(target_arch = "wasm32")]` — its API mirrors `wasmtime`, so `kernel.rs` would change little.

### Sandboxing & resource limits

`wasmtime` is a sandbox: no host memory access beyond the explicit imports, no syscalls. We
additionally:

- cap guest linear-memory growth per decode via `StoreLimits` (see `MAX_GUEST_MEMORY_BYTES`);
- validate every guest-returned pointer/length against the current memory size before reading;
- treat any guest trap or malformed Arrow C struct as a decode error (never a host panic).

CPU-time bounding (fuel or epoch interruption) is a planned follow-up — both are first-class in
`wasmtime`; epochs need a timer thread and fuel needs a budget, so the limit value is a policy
choice still to be made.

The kernel is untrusted data from the file, exactly like array bytes; correctness bugs in a
kernel can only corrupt *that array's* values, never host memory.

## Implementation phases

The first iteration used a bespoke `CanonicalMessage` wire format; it has been **replaced** by the
Arrow C Data Interface boundary with a dependency-free Rust guest. The remaining "next" work is the
on-disk change so the embedded blob is *only* the decoder over an otherwise-normal serialized array.

1. **Prototype (done, superseded):** `WasmKernel` over `wasmi` (since migrated to `wasmtime`),
   `WasmLayout`/`WasmReader`/`WasmLayoutStrategy`, `vx_decode_child`, and end-to-end round trips via
   WAT kernels and a bespoke `CanonicalMessage`. Proved the VM + layout + EOF kernel placement +
   child decode work end to end.
2. **Arrow C Data Interface import (done):** [`arrow_ffi::import`](../../vortex-wasm/src/arrow_ffi.rs)
   reconstructs a Vortex array (primitive + bool, incl. validity) from Arrow C structs in a guest
   memory image, via `from_arrow`.
3. **Arrow boundary, both directions (done):** [`arrow_ffi::export`](../../vortex-wasm/src/arrow_ffi.rs)
   writes host-decoded children as Arrow C structs into guest memory; `vx_decode` returns Arrow C
   structs; `CanonicalMessage` removed.
4. **Dependency-free Rust guest SDK (done):** `vortex-wasm-guest` builds/reads the C structs as
   plain bytes (no Arrow library, no nanoarrow, no Vortex crates), `#![no_std]` with an SDK-provided
   bump allocator; ~4 KB kernels. End-to-end tested against compiled fixtures in
   [`kernel_roundtrip.rs`](../../vortex-wasm/tests/kernel_roundtrip.rs).
5. **`WasmLayout` embeds only the decoder (next):** the strategy writes the encoded array in the
   existing serialized format (so a native VTable reads the same bytes without the blob) and embeds
   only the `.wasm`; the guest decodes from the serialized array flatbuffer it parses itself with
   `vortex-flatbuffers` (the generated code is pure `flatbuffers` + `alloc`). This replaces the
   current payload+child write model.
6. **Breadth (later):** `VarBinView`/`Struct`/`List` across the Arrow boundary, kernel dedup +
   caching, CPU-time limits (wasmtime fuel/epoch), and the `wasm32` fallback runtime if the browser
   reader needs WASM encodings.

Pushdown (filter/pruning into the kernel) is explicitly **out of scope** — WASM encodings only
decompress; the engine filters on the decoded output.

## Open questions

- **Kernel caching key:** digest of the blob vs. segment id; cross-file caching in a session.
- **Async vs. blocking:** running `wasmtime` on the IO runtime's blocking pool vs. a dedicated
  decode pool.
