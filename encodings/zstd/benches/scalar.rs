// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar reads out of a Zstd array. Cases are `(access_count, nullable, scattered, repeated)`;
//! clustered indices stay inside one frame so a retained decompression is reused, scattered ones
//! cross frames. A one-off read decompresses the frame holding the row every time, which is what
//! the repeated cases are measured against.

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;

fn main() {
    divan::main();
}

const LEN: usize = 16_384;
const VALUES_PER_FRAME: usize = 1024;
// The one-off cases stay short: without a retained decompression they cost one frame per access,
// so a thousand of them would dominate the shard's runtime without measuring anything else.
const CASES: &[(usize, bool, bool, bool)] = &[
    (1, false, false, false),
    (1, false, false, true),
    (64, false, false, false),
    (64, false, false, true),
    (1024, false, false, true),
    (1024, true, false, true),
    (1024, false, true, true),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

fn validity(nullable: bool) -> Validity {
    if nullable {
        Validity::from_iter((0..LEN).map(|i| i % 11 != 0))
    } else {
        Validity::NonNullable
    }
}

fn primitive(nullable: bool) -> ArrayRef {
    let input = PrimitiveArray::new(
        (0..LEN)
            .map(|i| u32::try_from(i / 16).vortex_expect("fixture values fit u32"))
            .collect::<Vec<_>>(),
        validity(nullable),
    );
    let mut ctx = SESSION.create_execution_ctx();
    Zstd::from_primitive(&input, 3, VALUES_PER_FRAME, &mut ctx)
        .vortex_expect("Zstd compression")
        .into_array()
}

fn var_bin(nullable: bool) -> ArrayRef {
    let input = VarBinViewArray::from_iter_nullable_str(
        (0..LEN).map(|i| (!nullable || i % 11 != 0).then(|| format!("value number {}", i / 16))),
    );
    let mut ctx = SESSION.create_execution_ctx();
    Zstd::from_var_bin_view(&input, 3, VALUES_PER_FRAME, &mut ctx)
        .vortex_expect("Zstd compression")
        .into_array()
}

fn indices(count: usize, scattered: bool) -> Vec<usize> {
    let span = if scattered { LEN } else { 256 };
    let base = if scattered { 0 } else { 4096 };
    let mut seed = 42u64;
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            base + ((seed >> 32) as usize % span)
        })
        .collect()
}

fn scalar_access(
    bencher: Bencher,
    array: ArrayRef,
    count: usize,
    repeated: bool,
    indices: &[usize],
) {
    bencher
        .with_inputs(|| {
            (
                SESSION.create_execution_ctx(),
                repeated.then(|| array.repeated_probe()),
                Vec::with_capacity(count),
            )
        })
        .bench_refs(|(ctx, probe, scalars)| {
            for &index in indices {
                let scalar = match probe {
                    Some(probe) => probe.execute_scalar(index, ctx),
                    None => array.execute_scalar(index, ctx),
                };
                scalars.push(scalar.vortex_expect("scalar access"));
            }
        });
}

#[divan::bench(args = CASES)]
fn primitive_scalar_access(
    bencher: Bencher,
    (count, nullable, scattered, repeated): (usize, bool, bool, bool),
) {
    let indices = indices(count, scattered);
    scalar_access(bencher, primitive(nullable), count, repeated, &indices);
}

#[divan::bench(args = CASES)]
fn var_bin_scalar_access(
    bencher: Bencher,
    (count, nullable, scattered, repeated): (usize, bool, bool, bool),
) {
    let indices = indices(count, scattered);
    scalar_access(bencher, var_bin(nullable), count, repeated, &indices);
}
