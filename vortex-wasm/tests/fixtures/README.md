<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# WASM kernel test fixtures

These `.wasm` files are decoder kernels compiled from the kernel crates that live alongside their
native encodings. They are committed so `tests/plugin_roundtrip.rs` and `tests/file_kernels.rs` can
exercise the full pipeline with real kernels (via `include_bytes!`) without building a
`wasm32-unknown-unknown` toolchain at test time.

| Fixture | Source crate | Encoding |
| --- | --- | --- |
| `bitpacked_kernel.wasm` | `encodings/fastlanes/wasm` | `fastlanes.bitpacked` |
| `fsst_kernel.wasm` | `encodings/fsst/wasm` | `vortex.fsst` |
| `runend_kernel.wasm` | `encodings/runend/wasm` | `vortex.runend` |
| `onpair_kernel.wasm` | `encodings/experimental/onpair/wasm` | `vortex.onpair` |

## Rebuilding

After changing the guest SDK or a kernel, rebuild and copy the fixtures:

```bash
(cd encodings/fastlanes/wasm && cargo build --target wasm32-unknown-unknown --release)
(cd encodings/fsst/wasm && cargo build --target wasm32-unknown-unknown --release)
(cd encodings/runend/wasm && cargo build --target wasm32-unknown-unknown --release)
(cd encodings/experimental/onpair/wasm && cargo build --target wasm32-unknown-unknown --release)
cp encodings/fastlanes/wasm/target/wasm32-unknown-unknown/release/vortex_fastlanes_wasm.wasm \
   vortex-wasm/tests/fixtures/bitpacked_kernel.wasm
cp encodings/fsst/wasm/target/wasm32-unknown-unknown/release/vortex_fsst_wasm.wasm \
   vortex-wasm/tests/fixtures/fsst_kernel.wasm
cp encodings/runend/wasm/target/wasm32-unknown-unknown/release/vortex_runend_wasm.wasm \
   vortex-wasm/tests/fixtures/runend_kernel.wasm
cp encodings/experimental/onpair/wasm/target/wasm32-unknown-unknown/release/vortex_onpair_wasm.wasm \
   vortex-wasm/tests/fixtures/onpair_kernel.wasm
```

The onpair kernel needs no extra flags: its crate carries a `.cargo/config.toml` selecting
`getrandom`'s custom backend, which `onpair`'s training-only `rand` dependency would otherwise fail
to build for `wasm32-unknown-unknown`.
