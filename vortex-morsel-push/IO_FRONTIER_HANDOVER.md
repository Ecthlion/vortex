# DuckDB push-frontier SF10 handover

Status: 2026-09-14. This is the canonical handover for the current production-integration
experiment: DuckDB TPC-H SF10 on the 14-logical-CPU Apple Silicon host, comparing the established
V1 scan with push-frontier plus the opt-in W8 adjacent-morsel policy. The full retained evidence is
`/private/tmp/duckdb-thread-bridge-v1-w8-full22.ahG9bN`.

The older DataFusion and direct-harness work remains useful implementation history, but it is not
the acceptance result for the current DuckDB path. See [Historical evidence](#historical-evidence)
and [PUSH_FRONTIER_BENCHMARK_CHECKLIST.md](PUSH_FRONTIER_BENCHMARK_CHECKLIST.md).

> **Exact-current validation transcript:**
> `/private/tmp/duckdb-frontier-final-validation.8napci` contains every final command, complete
> stdout/stderr, and exit status for the checkpoint worktree. That artifact validates tests,
> formatting, checks, and lints; it does not replace the frozen build, smoke, correctness, plan, and
> performance artifacts, and no fresh performance campaign was run during final validation.

## Executive summary

- W8 is a real HOT, all-core time win over V1: Q1-Q22 sum of medians is `0.897432x`, and the
  equal-query/all-pair geometric mean is `0.876156x` with bootstrap 95% CI
  `[0.846938, 0.903788]` and `72/88` paired wins.
- The strongest query wins are Q12 (`0.5356x` paired geometric mean), Q6 (`0.6532x`), Q20
  (`0.7703x`), and Q14 (`0.7984x`). No query meets the frozen credible-regression rule of median
  ratio above `1.05` and paired-bootstrap lower bound above one.
- W8 does **not** pass the resource acceptance gate. Equal-query RSS is `1.140203x`; Q8 reaches
  `1.528545x`, Q10 adds `216.68 MiB`, the maximum paired child reaches `1.590897x` and
  `+239.47 MiB`, and `16/88` children violate the hard `1.50x` / `+96 MiB` bound. The mechanism is
  retained as promising opt-in work, not approved as a production default.
- W8 means bundle size two with a minimum of eight output waves per DuckDB thread. It applies only
  to filtered external DuckDB push-frontier scans whose `MorselScan` carries no limit. Its `N` is
  the selected pre-static-prune morsel count after row-range/Selection and applicable executor-limit
  truncation; it activates when `ceil(N / 2) >= threads * 8`. It pairs contiguous selected morsels
  before static pruning. After pruning, the pair remains one output task; only cross-morsel predicate
  down lookahead can fall back from one to zero. Projection remains demand-gated.
- The normal push-frontier policy remains down lookahead `0`, two predicate-only right groups for
  the current range, and refill size `32`. `push` remains the established eager policy. V1 and
  plain push are unchanged by W8.
- All `176/176` measured V1/W8 children produced the same strict canonical result, and all 22 raw
  V1/W8 EXPLAIN pairs were byte-identical. Q15's multi-statement path also passed.
- Evidence is HOT local-file evidence only. Cold cache, remote object stores, generic naming,
  LIMIT behavior, default enablement, and the remaining memory attribution are open.

## Frozen source, binary, and data identity

| Item | Frozen value |
|---|---|
| Branch | `ji/push-frontier-all-bench` |
| Source HEAD | `8295653589d85fe6e8bd12824eff9b6687760fc9` |
| Tracked working-tree patch SHA-256 | `a1031e0aacff32ec094696332cbc01cb57723ac8fbc0d6052fb99de9dfee80c2` |
| Untracked `diagnostic_read_at.rs` SHA-256 | `e1375ed68116268bb9f02507acebdaf3119463fbad3c4d74ed535af1babe422d` |
| Generated DuckDB binding snapshot SHA-256 | `844dda5a97f9eb83079c791a0d9c084ee2e44dd0ba573b60fb836e6eaa6fb393` |
| Frozen binary | `/private/tmp/duckdb-thread-bridge.bL8KwF/bin/duckdb-bench` |
| Binary SHA-256 | `3802155325e1f8de58ac1aa08ca11b1a7a410b226e6d7e34e987a795bcacdc02` |
| Binary artifact manifest SHA-256 | `ae1c0b8fd5101aec49e41e69b68c6fec8c6edfc2dd4c8cf80a1a8878a673bc0f` |
| Mach-O UUID | `7C32740F-A8F4-304E-AA79-3DE1A3281CFC` (`arm64`) |
| `Cargo.lock` SHA-256 | `5ccc1ce20bd67b916ee90e6114708cde8658b18df1d832437bc0d29898f00412` |
| `rust-toolchain.toml` SHA-256 | `af53ffeb45c33de4be1c5cb84557a907c6011bd7ff21bd9f33ae4bf1a1719f34` |
| Canonical campaign manifest SHA-256 | `3dd2690caf74d47efe1e66d180a66549e89a513fbc2a717fdcdf7b798a4951db` |
| Canonical campaign `SHA256SUMS` SHA-256 | `d35ffdcd054db36a624a85eb4508785979c9a9934153f196f1583e74c3fb2463` |

The frozen source was deliberately a dirty worktree, not a commit: 32 tracked modifications plus
the untracked ObjectStore diagnostic source before these three documentation edits. The binary
artifact copied the tracked patch, untracked source, and generated binding snapshot into
`/private/tmp/duckdb-thread-bridge.bL8KwF/meta`. The patch contained the thread bridge, strict
DuckDB result/EXPLAIN support, push-frontier predicate-only policy, W8 mechanism, and other
in-progress diagnostics. The canonical campaign recorded identity before every block and after
every plan block. Those snapshots match the binary, source patch, lockfile, toolchain, all TPC-H SQL
files, and all input files. Documentation edits made after this handover do not change that frozen
binary identity.

The canonical data is the real TPC-H SF10 Vortex set plus DuckDB metadata:

| Input | SHA-256 |
|---|---|
| `customer.vortex` | `cb63fe0071ca3719a70b390f31e4ba1e4cf1d9b63ffc4460194b2853cf37a544` |
| `lineitem.vortex` | `b547ba17aa045a1ad535392556832ba4e75ebecb841173631ddedff41743187e` |
| `nation.vortex` | `2b2b607517211aae7909e659a552ed47f837af06a5e24f39892c86e41074a9f4` |
| `orders.vortex` | `4777103e8ba91441e2487b040dc4c44e656cd43d4ef003de052b5c3414cdb66e` |
| `part.vortex` | `b783ea6375c00478632890bbc84f34f243ff0efe6c294b242ddd23585e6dd35f` |
| `partsupp.vortex` | `4c8d69ea184ec56468a26cbf7db820a88c42d31ce4fcd2bbd2da453c77503cd7` |
| `region.vortex` | `46811ae9461a16f15824fab79f71acdadebf8235777c6f0932f1eec5ec33f144` |
| `supplier.vortex` | `80a1ddc14ef4fe7cef54d386430e9450e2267e700819dce6da2778ecb08eda5e` |
| `duckdb.db` | `7002f710f48dc809daca74710d4fc8cd690cbc147284524941da1da0c5e66bf0` |

All 22 SQL hashes are in
`/private/tmp/duckdb-thread-bridge-v1-w8-full22.ahG9bN/meta/pre-run-hashes.json`; the campaign's Q6
hash is `896340eecc214dd73cd411d71d9938e3138add1b49cb1d455d80416426f755b3`. The checked-in benchmark
source is `vortex-bench/sql/tpch/q1.sql` through `q22.sql`; Q15 is intentionally multi-statement.

## Fairness and HOT protocol

Both arms used the same frozen binary, real SF10 files, canonical TPC-H SQL, default fields,
DuckDB `--threads 14`, one iteration, fresh processes, no reuse, and strict deterministic result
emission. Only the scan backend and W8 opt-in differed:

- V1: `VORTEX_SCAN_BACKEND=v1`; all bundle variables unset.
- W8: `VORTEX_SCAN_BACKEND=push-frontier`, `VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE=2`; min-waves unset,
  selecting the compiled default of eight.

Diagnostics, profiling, tracing, oracle mode, extension override, scan-API override, memory tracker,
optional-filter override, and the obsolete bundle-thread hint were unset. `RUST_LOG=warn`.

The four retained measured passes were:

1. Q1-Q22, V1 then W8.
2. Q22-Q1, W8 then V1.
3. Q1-Q22, W8 then V1.
4. Q22-Q1, V1 then W8.

Every measured child immediately followed an identical-arm/query fresh-process prewarm. The
campaign retained all `176` measured children, all `176` excluded prewarms, and `44` plan children.
No performance sample was rerun, dropped, or classified as an outlier. All `396` pre-child process
checks found no concurrent build, benchmark, or profiler. AB/BA geometric means are `0.878651` and
`0.873669`, a `0.57%` spread.

Internal materialized-query duration is the primary performance metric. Supervisor wall time and
resource values include process startup, query setup, strict post-timing result rendering, and
teardown. This is a HOT protocol: no cache eviction was attempted, and nothing here is a cold-cache
or remote-store result.

The frozen gates were:

- global strong win: ratio at most `0.95`; a 30%-faster claim requires at most `0.70`;
- credible query regression: median W8/V1 above `1.05` and paired-bootstrap lower bound above `1`;
- AB/BA order-stratum spread at most `5%`;
- per-query median RSS at most `1.30x` and `+64 MiB`;
- hard child RSS at most `1.50x` and `+96 MiB`.

## Full V1 versus W8 result

Ratios below one favor W8. `W8 active/scan` is contextual activation inventory from the previous
diagnostic binary; timing itself ran with diagnostics off. The post-thread-bridge Q6 diagnostic
separately proved configured parallelism `14` and the same W8 bundle plan.

| Q | W8 active/scan | V1 ms | W8 ms | Paired G [95% CI] | Wins | Median ratio | RSS MiB V1/W8 (ratio) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 174.455 | 178.003 | 1.0347 [1.0158,1.0616] | 0/4 | 1.0203 | 221.8/248.7 (1.121) |
| 2 | 0/9 | 37.673 | 39.013 | 1.0368 [1.0150,1.0591] | 0/4 | 1.0356 | 214.9/221.8 (1.032) |
| 3 | 1/3 | 130.321 | 114.863 | 0.8842 [0.8610,0.9080] | 4/4 | 0.8814 | 295.2/319.1 (1.081) |
| 4 | 1/2 | 121.717 | 105.094 | 0.8592 [0.8536,0.8670] | 4/4 | 0.8634 | 267.2/295.9 (1.107) |
| 5 | 1/6 | 154.905 | 141.909 | 0.9144 [0.8961,0.9271] | 4/4 | 0.9161 | 396.9/469.2 (1.182) |
| 6 | 1/1 | 57.817 | 37.873 | 0.6532 [0.6473,0.6571] | 4/4 | 0.6551 | 119.5/127.4 (1.067) |
| 7 | 1/6 | 138.730 | 117.200 | 0.8437 [0.8286,0.8598] | 4/4 | 0.8448 | 344.6/360.7 (1.047) |
| 8 | 1/8 | 165.910 | 156.583 | 0.9356 [0.9238,0.9480] | 4/4 | 0.9438 | 300.1/458.7 (1.529) |
| 9 | 1/6 | 348.427 | 345.325 | 0.9918 [0.9883,0.9952] | 4/4 | 0.9911 | 791.0/941.6 (1.190) |
| 10 | 1/4 | 180.669 | 173.126 | 0.9600 [0.9352,0.9869] | 3/4 | 0.9582 | 1117.2/1333.8 (1.194) |
| 11 | 0/4 | 31.909 | 28.413 | 0.8955 [0.8776,0.9134] | 4/4 | 0.8904 | 226.5/234.6 (1.036) |
| 12 | 1/2 | 135.636 | 72.776 | 0.5356 [0.5263,0.5451] | 4/4 | 0.5366 | 235.1/266.7 (1.134) |
| 13 | 0/2 | 214.893 | 188.979 | 0.8811 [0.8698,0.8904] | 4/4 | 0.8794 | 615.5/692.9 (1.126) |
| 14 | 1/2 | 82.505 | 65.557 | 0.7984 [0.7930,0.8057] | 4/4 | 0.7946 | 273.3/298.1 (1.091) |
| 15 | 1/2 | 86.505 | 69.704 | 0.8077 [0.7970,0.8204] | 4/4 | 0.8058 | 278.0/320.0 (1.151) |
| 16 | 0/3 | 48.451 | 48.600 | 0.9959 [0.9817,1.0073] | 2/4 | 1.0031 | 399.8/400.7 (1.002) |
| 17 | 1/3 | 121.680 | 105.489 | 0.8656 [0.8558,0.8835] | 4/4 | 0.8669 | 166.2/218.0 (1.311) |
| 18 | 0/4 | 228.828 | 223.994 | 0.9781 [0.9632,0.9992] | 3/4 | 0.9789 | 1013.4/1145.6 (1.130) |
| 19 | 1/2 | 101.976 | 90.680 | 0.8944 [0.8855,0.9040] | 4/4 | 0.8892 | 336.4/329.3 (0.979) |
| 20 | 1/5 | 125.495 | 102.157 | 0.7703 [0.6784,0.8321] | 4/4 | 0.8140 | 236.5/306.3 (1.295) |
| 21 | 3/6 | 421.907 | 380.571 | 0.9018 [0.8957,0.9080] | 4/4 | 0.9020 | 702.3/780.7 (1.112) |
| 22 | 0/2 | 47.277 | 47.898 | 1.0446 [1.0108,1.1012] | 0/4 | 1.0131 | 169.6/220.2 (1.298) |

### Global time and resource summary

| Metric | Equal-query geometric ratio | Sum-of-medians ratio | Interpretation |
|---|---:|---:|---|
| Internal query time | 0.876156 | 0.897432 | Primary; W8 wins |
| Supervisor real time | 0.930969 | 0.934807 | Includes startup/strict-output overhead |
| User CPU time | 0.974167 | 0.979072 | Slight reduction |
| System CPU time | 1.510153 | 1.491097 | Large regression |
| Instructions | 1.038787 | 1.023740 | More retired work |
| Cycles | 1.088106 | 1.061472 | More cycles despite lower wall time |
| Involuntary context switches | 2.394633 | 2.320915 | Large scheduling increase |
| Maximum RSS | 1.140203 | 1.145528 | Resource gate fails |
| Peak footprint | 1.155063 | 1.156252 | Resource gate fails |

The speed result passes the frozen global time gate and misses the 30%-faster threshold. Resource
acceptance fails plainly; speed does not override that failure.

## B1 versus W8: what W8 itself contributes

`/private/tmp/duckdb-w8-full22.0wO1Zw` compared bundle-one push-frontier (B1) with W8 in the same
pre-thread-bridge frozen binary. This isolates the adjacent-morsel policy better than V1/W8 does.

| Comparison | B1 | W8 | W8/B1 or evidence |
|---|---:|---:|---:|
| Sum of Q1-Q22 medians | 2855.974 ms | 2835.298 ms | 0.9928 |
| Equal-query/all-pair geometric mean | - | - | 0.9879 |
| Q6 paired geometric mean | 42.332 ms median | 37.348 ms median | 0.8826 `[0.8748,0.8918]`, 4/4 |
| Q19 paired geometric mean | 95.533 ms median | 91.906 ms median | 0.9613 `[0.9552,0.9673]`, 4/4 |
| Same-policy queries Q2/Q11/Q13/Q16/Q18/Q22 | - | - | 0.9968 |
| Maximum RSS equal-query geometric mean | - | - | 1.0695 |
| Supervisor wall / user / system | - | - | 0.9874 / 0.9908 / 0.9859 |
| Instructions / cycles / involuntary CS | - | - | 0.9898 / 0.9846 / 0.9538 |

This campaign passed its predeclared B1/W8 mechanism gates and found no credible greater-than-5%
regression. It does not replace the later V1/W8 resource verdict, and its activation inventory came
from the previous binary. Manifest SHA-256 is
`d1142ae7cc79c6e26d13705afebd77a250372f76fe78617bfc8c67f1ecc9bab3`; `SHA256SUMS` SHA-256 is
`2299ed2a24f192e0a6bfa1646024021be0a1d41e0f62aeafa6606db4dcbc8f94`.

## Decision ledger

| Candidate | Decision | Exact evidence and reason |
|---|---|---|
| Thread bridge | Retain | The DuckDB scheduler's configured thread count reaches Rust scan initialization. Post-bridge Q6 W8/B1 was `0.8842818`, CI `[0.8717194,0.8971873]`, 8/8. Artifact `/private/tmp/duckdb-thread-bridge-q6-ab.Dm5bhd`; manifest `51cda179e7a78c9dcdec2fc15c2f789606e2a331dba0f30d5dda51aa9bddf619`, sums `ed26e634f9bb1ead421f26b234c385f297de22cd0c5ce6b669cfb5b1e5c08c27`. |
| SourceRegistered / prepared projection | Reject | Q6 candidate/parent PF geometric mean `1.02157`, CI `[1.01092,1.03111]`, 1/8 wins; RSS `121.39 -> 137.63 MiB` (`1.1338x`). It lengthened the wait-critical path without a sampled allocator/lock win. Artifact `/private/tmp/duckdb-q6-source-reg-ab.v5B3rR`; manifest `d94d8a72658e8f0a930deb7260254a755c36647f187306662c4aa96a0af53498`, sums `949f39cefdfc8c8e24f3c3c4edd394fd025acd4dd34cc7c6fbbfb97deccd1b7c`. Removed; do not revive as a passive hint without new evidence. |
| Unconditional B2/W1 | Reject globally | Full-suite sum ratio `0.99824`, query-median geometric mean `1.00189`, all-pair `1.00314`; Q2 `1.0875` and Q11 `1.1771`, both 0/4 with CI above one. RSS sum `1.0757`, footprint `1.0813`. Artifact `/private/tmp/duckdb-tpch22-bundle2-perf.u6Q4Ug`; manifest `73559bcea0dda281412fc1d2082ec062c75bccb46bbdf03082a60a0db85e372b`, sums `64db18af1b937fe504cd8c84984968679074b854a422501db904ce2a841ea8a4`. |
| Required/static-filter gate | Reject mechanism | The attempted filter classifier changed no activation because every `additional_filters` entry was treated as required; DuckDB's optional/dynamic filter is a recursive filter kind, not distinguishable by merely testing whether the set is nonempty. Timing was neutral but did not validate the policy. Artifact `/private/tmp/duckdb-required-filter-ab.x8cdpa`; manifest `cb80a196a61eaf1b94d453c3729424dcda7955d31a3ce10413e14a33e29fdad9`, sums `d5a710d356ddce5da9f89e3c70c30ee6ff04d5a31e74e05a0d98cd0f939f5112`. |
| W8 occupancy gate | Retain as opt-in | Replaces unconditional B2 with `ceil(N/2) >= 8P`, where `N` is the selected pre-static-prune count after row-range/Selection and applicable executor-limit truncation. B1/W8 mechanism screen passes; final V1/W8 time result is strong. The final resource gate fails, so it is not ready to become the default. |
| DataFusion weak raw-source sharing + bounded gap | Retained, DataFusion-only | The dirty patch retains a weak, identity-keyed raw-source pool and bounded aggregate gap/coalescing work in the DataFusion path. This is separate from SourceRegistered and is not exercised by the canonical DuckDB binary path. Mechanism captures motivated it, but there is no accepted same-binary production timing result for the retained combination. |
| Persistent local `File` payload promotion | Parked/unaccepted | The ObjectStore/DataFusion persistent-file promotion WIP aims to reuse a local file payload for positional reads. It is not part of W8, has no accepted performance campaign, and must not be described as retained production behavior merely because code remains in the dirty worktree. |

## Complete activation and morsel-shape inventory

This table is the non-timed inventory from `/private/tmp/duckdb-w8-full22.0wO1Zw`. Natural morsel
counts are grouped as `{morsels: number_of_scans}`. Because this artifact used full selections,
those natural-morsel values equal the selected pre-static-prune `N` after row-range/Selection and
applicable executor-limit truncation; that equality is artifact-specific, not a general gate label.
Activation requires an external PushFrontier scan, an executable filter, no
`MorselScanBuilder` limit, and the occupancy threshold. Pairability is evaluated afterward, so an
active B2 scan may yield zero pairs. Within an active scan, only selected morsels whose coverage is
one range each and exactly adjacent are paired. `Same as B1` means W8 did not activate.

| Q | Scans | Natural morsels | Active scans | Active morsels | Output/paired bundles | Same as B1 |
|---:|---:|---|---:|---|---:|---|
| 1 | 1 | `{458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 2 | 9 | `{1: 6, 16: 1, 62: 2}` | 0 | `{}` | 0/0 | yes |
| 3 | 3 | `{12: 1, 115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 4 | 2 | `{115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 5 | 6 | `{1: 3, 12: 1, 115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 6 | 1 | `{458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 7 | 6 | `{1: 3, 12: 1, 115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 8 | 8 | `{1: 4, 12: 1, 16: 1, 115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 9 | 6 | `{1: 2, 16: 1, 62: 1, 115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 10 | 4 | `{1: 1, 12: 1, 115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 11 | 4 | `{1: 2, 62: 2}` | 0 | `{}` | 0/0 | yes |
| 12 | 2 | `{115: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 13 | 2 | `{12: 1, 115: 1}` | 0 | `{}` | 0/0 | yes |
| 14 | 2 | `{16: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 15 | 2 | `{1: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 16 | 3 | `{1: 1, 16: 1, 62: 1}` | 0 | `{}` | 0/0 | yes |
| 17 | 3 | `{16: 1, 458: 2}` | 1 | `{458: 1}` | 229/229 | no |
| 18 | 4 | `{12: 1, 115: 1, 458: 2}` | 0 | `{}` | 0/0 | yes |
| 19 | 2 | `{16: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 20 | 5 | `{1: 2, 16: 1, 62: 1, 458: 1}` | 1 | `{458: 1}` | 229/229 | no |
| 21 | 6 | `{1: 2, 115: 1, 458: 3}` | 3 | `{458: 3}` | 687/687 | no |
| 22 | 2 | `{12: 1, 115: 1}` | 0 | `{}` | 0/0 | yes |

The diagnostic plan events did not carry a stable file/scan identity; their order is retained as
text-order ordinals, not claimed as a physical scan mapping. This is why activation counts are
context, not performance evidence.

## Architecture and ownership

DuckDB owns the outer parallel task schedule. Its global scan state holds file/range splits; each
DuckDB local state claims one split under the file lock and scans outside that lock. The thread
bridge supplies DuckDB's configured scheduler thread count to Rust when scan state is initialized.
It does not create another `P`-wide executor.

For the external push-frontier path, each claimed split constructs a one-worker `MorselScan` over
that split's range or W8 pair. The scan uses one file-scoped `IoService` and segment source; the
external owner drives output and is responsible for service lifetime:

```text
DuckDB global scan state
  -> one split claimed by a DuckDB local state
  -> external one-worker MorselScan for one natural range or one W8 pair
  -> file-scoped IoService / FileSegmentSource
  -> ObjectStoreReadAt local File payload
  -> blocking-pool positional read
  -> combined bundle retained by the local exporter until DuckDB consumes its chunks
```

The executor first derives selected natural morsels from the row range/Selection and applies any
executor-limit truncation. W8 uses that pre-static-prune count for its occupancy decision and pairs
two adjacent morsels only when their selected coverage is each a single range and exactly
contiguous. Static pruning then runs independently inside each already-created bundle. Pruning does
not debundle the task: gaps, empty members, or multi-range survivors retain the pair but force
cross-morsel down lookahead to `0`. Down lookahead is `1` only when both post-prune members retain
exactly one range and those ranges remain exactly adjacent. Odd pre-prune tails remain single
tasks. These rules preserve disjoint ownership without duplicate or missing rows. One external scan
worker means W8 does not park a second internal worker or multiply the outer DuckDB thread count.

Current source anchors in the frozen patch are:

| Area | Source |
|---|---|
| Push-frontier down/right/refill defaults and bundle size | `vortex-morsel-push/src/executor.rs:74-90` |
| Bundle eligibility, selected-morsel threshold, and pairing | `vortex-morsel-push/src/executor.rs:707-927` |
| External one-worker driver | `vortex-morsel-push/src/driver.rs:4233-4295` |
| DuckDB opt-in parsing and W8 default | `vortex-duckdb/src/file_reader.rs:114-198` |
| Thread propagation into scan construction | `vortex-duckdb/src/file_reader.rs:327-377` |
| Split claim and exporter lifetime | `vortex-duckdb/src/file_reader.rs:388-431` |
| DuckDB C++ scheduler bridge | `vortex-duckdb/cpp/multi_file_reader.cpp:155-163` and `vortex-duckdb/cpp/include/table_function.h:25-33` |
| Rust thread bridge | `vortex-duckdb/src/duckdb/table_init_input.rs:38-47` and `vortex-duckdb/src/table_function.rs:247-308` |
| Strict result/EXPLAIN CLI | `benchmarks/duckdb-bench/src/main.rs:107-116` and `benchmarks/duckdb-bench/src/main.rs:193-304` |
| Canonical typed multiset rendering | `benchmarks/duckdb-bench/src/lib.rs:350-417` |

### Same-morsel right speculation versus cross-morsel down speculation

These are separate policies:

- Predicate-only right lookahead exposes up to the next two conjunct groups for the **current**
  morsel. It lets the source know adjacent predicate dependencies early but does not expose the
  projection group. The current production-integration default is right `2`, down `0`, refill `32`.
- W8 decides and pairs adjacent selected natural morsels before static pruning. Within the retained
  pair, down `1` exposes predicate groups for the second morsel only if both post-prune members have
  exactly one range and remain exactly adjacent; otherwise the pair still executes as one task with
  down `0`. Projection for either morsel remains demand-gated.
- Refill size is an admission cadence, not permission to speculate arbitrary ranges. W8's
  `ceil(N/2) >= 8P` gate ensures at least eight outer work waves per DuckDB thread after pairing.

The rejected `1/0/32` experiment increased down lookahead without the external adjacent-range
ownership needed to make that second range useful. The rejected 512-Ki-row morsel experiment
changed work-unit size globally. Neither is equivalent to bounded two-range external bundling and
neither should be repeated as evidence for or against W8.

## I/O, coalescing, locks, and scheduler findings

- A local ObjectStore child calls `get_opts`; LocalFileSystem returns a `File` payload; the Vortex
  adapter dispatches aligned positional reads to its reusable blocking pool. This is not one pthread
  creation per read. The available instrumentation does not prove exact kernel `open`, `fstat`, or
  `pread` syscall counts.
- `FileSegmentSource` can coalesce only dependencies visible together. Predicate-only right
  lookahead exposes same-morsel predicate segments; W8 adds the adjacent morsel's predicate
  dependencies. Projection is intentionally absent until the predicate gate requires output.
- The B2 mechanism diagnostic changed Q6 from `1141` physical calls / `1463` ranges to `848` calls /
  `1097` ranges, kept the physical union equal within `556` bytes, reduced overlap by `61.4%`, and
  added `11.53 MiB` RSS. That run was instrumented and is mechanism evidence only, not timing
  evidence.
- Older intrusive captures reported different absolute shapes, including `636` calls / `1286`
  ranges / mean batch `2.022` / maximum active `23`. Their tracing volume and construction/poll
  instrumentation materially changed batching. Do not compare those absolute counts with the
  diagnostic-off performance campaign.
- There is no evidence that a single global mutex is the primary bottleneck. The DuckDB file lock
  covers split claim, not scan execution. SourceRegistered profiles showed less, not more, sampled
  lock/allocator work while wall time regressed. Historical DataFusion metrics also found zero
  measured lock contention.
- Higher system time and involuntary context switches in the final V1/W8 run are real whole-process
  signals, but current evidence does not assign them uniquely to pread, blocking-pool queueing,
  DuckDB scheduling, or exporter backpressure.

## Samply evidence

The SourceRegistered investigation is in
`/private/tmp/duckdb-q6-source-registered-samply-20260913.9KAQ6V`:

```bash
samply load \
  /private/tmp/duckdb-q6-source-registered-samply-20260913.9KAQ6V/candidate/profile/profile.json.gz
samply load \
  /private/tmp/duckdb-q6-source-registered-samply-20260913.9KAQ6V/parent/profile/profile.json.gz
```

The candidate query window was `73.4 ms` versus parent `68.4 ms`, with effective CPU occupancy
`8.76` versus `9.84`. Worker parking/channel stacks added roughly `62-64 ms` inclusive in the
candidate. Object-store/read, locks, allocation, and source-driver CPU were lower, and no
SourceRegistered/Prepare stack was sampled. There was one profile per arm; sampler slowdown was
approximately `1.525x` candidate and `1.296x` parent, and the profile timing rank reversed the
non-profile result. Use it only to locate wait-path changes, never as performance acceptance.

## Memory attribution and lifetime

The final full-suite process-level result is unambiguous: W8 versus V1 fails the RSS/footprint
gate. Attribution is only partial.

Across the earlier pre-thread-bridge B1/W8 campaign and the final V1/W8 campaign, approximate RSS
decomposition is:

| Q | V1 MiB | B1 MiB | W8 MiB | Approx. V1->B1 | Approx. B1->W8 |
|---:|---:|---:|---:|---:|---:|
| 8 | 300.102 | 398.008 | 460.211 | +97.906 | +62.203 |
| 10 | 1117.164 | 1306.695 | 1312.305 | +189.531 | +5.609 |
| 17 | 166.203 | 192.148 | 218.898 | +25.945 | +26.750 |
| 20 | 236.500 | 262.586 | 302.266 | +26.086 | +39.680 |
| Global sum | 8720.859 | 9478.953 | 9999.383 | 1.08693x | 1.05490x |

This decomposition is approximate because the B1/W8 and V1/W8 binaries straddle the thread
bridge. The final current-W8 global sum is `9989.992 MiB`, close but not identical to the earlier
W8 sum.

The clean same-binary B1/W8 memory screen at `/private/tmp/duckdb-b1-w8-memory.NPeKjU` isolates
the bundle increment more directly:

| Condition | Q8 W8-B1 RSS | Q10 W8-B1 RSS | W8 activation context |
|---|---:|---:|---|
| SF1/P14 | +9.66 MiB | +2.89 MiB | W8 inactive |
| SF10/P14 | +80.53 MiB | +25.85 MiB | one 458-morsel scan active |
| SF10/P1 | +8.72 MiB | +0.29 MiB | more scans eligible, one DuckDB worker |

This supports a large thread/concurrency component, not a claim that the whole-process delta is
constant. Manifest SHA-256 is
`49834b11b01d632f02ce931a9d858c6ceee3df3f73af68fe4c5cec8fe3ba42fa`; sums SHA-256 is
`e5c823639b2936b0fee8c184eb5bce665585ca21e9743ac1d2404033b7c382f5`.

The same-binary scaling artifact `/private/tmp/duckdb-w8-memory-scaling.IqJzs2` is mixed:

| Condition | Q | Active W8 scans | V1 MiB | W8 MiB | Delta |
|---|---:|---:|---:|---:|---:|
| SF1/P14 | 8 | 0 | 225.62 | 236.13 | +10.52 |
| SF1/P14 | 10 | 0 | 271.84 | 278.94 | +7.09 |
| SF10/P14 | 8 | 1 | 305.56 | 471.66 | +166.09 |
| SF10/P14 | 10 | 1 | 1106.28 | 1336.09 | +229.80 |
| SF10/P1 | 8 | 3 | 155.35 | 228.04 | +72.69 |
| SF10/P1 | 10 | 2 | 915.41 | 886.46 | -28.95 |

This rejects a simple data-linear explanation and shows strong concurrency sensitivity, but the
policy activates different scans across scale/thread conditions, so it does not prove a clean
constant-memory law. Manifest SHA-256 is
`f75990fb4a9e4b5392b4ec75f8cb8a406ba218e2ab2010ec39f437fbf9c8b76a`; sums SHA-256 is
`fc6de5b18b02a6adc5ef73a88efafcdcfe1f84c80e792378d1cc7f5fbc90de95`.

The intended live bound is active DuckDB local states times bundle size two times schema/output
width. Each local exporter retains one returned combined bundle until DuckDB consumes its chunks.
Raw cells, decoded cells, split/plan metadata, DuckDB operators, and the file-scoped
`ConversionCache` can add to that. Plan metadata, segment catalogs, and split descriptors may scale
with file size even if live decoded/output buffers remain window-bounded. `ConversionCache`
lifetime and bytes have not been independently measured.

Metrics needed to close attribution are low-cardinality per-scan totals for selected natural
morsels, paired bundles, maximum live exporters, rows/bytes retained by exporters, raw and decoded
cell high-water bytes, `ConversionCache` entries/bytes, blocking queue high water, active reads, and
completed/cancelled reads. Do not use range/path labels in registry metrics.

## Correctness, plans, and default-field equivalence

Strict DuckDB output mode runs after the timed query and is default-off. It emits deterministic
column names, DuckDB logical types, null markers, and escaped text values, sorts complete encoded
rows lexicographically, and preserves duplicates. Query errors and expected-row-count mismatches
fail the process in strict mode.

For the canonical campaign:

- all `176/176` measured children matched their frozen schema/type/value/multiset hash;
- every child emitted one consistent GH record and ingest record, exited zero, and had empty
  stderr;
- all 22 raw V1/W8 EXPLAIN pairs were byte-identical;
- normalized physical operators, filters, projections, and `READ_VORTEX` counts matched;
- both arms used canonical SQL, identical Vortex files, the same default projected fields, and
  default selection; `VORTEX_USE_SCAN_API` was unset;
- Q15's multi-statement create/query/drop path passed.

This is a public DuckDB result gate, not an internal scan-row comparison. The renderer is exact for
the DuckDB text representation it receives; it has not been proven to preserve every possible
scalar payload bit pattern across all DuckDB logical types. That limitation does not invalidate the
TPC-H types exercised here, but it must be tested before calling the helper a universal bit-exact
artifact format.

## Exact reproduction commands and environment

Prefer analyzing or resuming the frozen artifact over rebuilding. A rebuild creates a new identity
and must not be mixed with these samples.

Build a new candidate once:

```bash
env RUSTC_WRAPPER= cargo build -p duckdb-bench \
  --profile release_debug --features unstable_encodings
```

The canonical runner's base query command was structurally:

```bash
BIN=/private/tmp/duckdb-thread-bridge.bL8KwF/bin/duckdb-bench
ROOT=/private/tmp/duckdb-thread-bridge-v1-w8-full22.ahG9bN
QUERY=6

/usr/bin/time -l -o "$ROOT/raw/manual-v1.time" \
  /usr/bin/env \
  -u VORTEX_SCAN_DIAGNOSTICS \
  -u VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE \
  -u VORTEX_DUCKDB_FRONTIER_BUNDLE_MIN_WAVES \
  -u VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE_THREADS \
  -u VORTEX_DUCKDB_FRONTIER_BUNDLE_ALLOW_OPTIONAL_FILTER \
  -u VORTEX_USE_SCAN_API \
  RUST_LOG=warn VORTEX_SCAN_BACKEND=v1 \
  "$BIN" tpch --opt scale-factor=10.0 --formats vortex \
  --queries "$QUERY" --iterations 1 --threads 14 --hide-progress-bar --strict \
  --runner manual-v1 --emit-results --display-format gh-json \
  -o "$ROOT/raw/manual-v1.gh.jsonl" \
  --ingest-jsonl "$ROOT/raw/manual-v1.ingest.jsonl"
```

For W8, use the same command and change only:

```bash
VORTEX_SCAN_BACKEND=push-frontier \
VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE=2
```

Leave `VORTEX_DUCKDB_FRONTIER_BUNDLE_MIN_WAVES` unset to select W8. The campaign also explicitly
unset profiling, tracing, oracle, extension, memory-tracker, and obsolete hint variables listed in
its `run_campaign.py`; use that runner's `UNSET` list rather than treating the abbreviated manual
template as the full campaign protocol. A fair measured child must be preceded immediately by an
identical fresh-process prewarm and must follow the four-pass AB/BA order above. Do not add
`VORTEX_SCAN_DIAGNOSTICS`, Samply, or result-oracle work to a timing child.

`VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE_THREADS` was the historical manual thread hint and is no longer
read by the current source; it was still explicitly unset to prove it did not influence the frozen
campaign. `VORTEX_DUCKDB_FRONTIER_BUNDLE_ALLOW_OPTIONAL_FILTER` belonged to the failed filter-gate
experiment and was likewise unset. Do not substitute similarly named variables when reproducing
the historical environment.

To validate plans, append `--explain`; preserve stdout/stderr and compare both raw rows and
normalized scan/filter/projection inventory before timing.

## Tests, checks, and provenance caveat

The mechanism work added focused coverage for thread propagation, split ownership, adjacent/disjoint
and odd-tail pairing, pre-prune W8 activation/threshold behavior, retained post-prune bundles with
down-`0` fallback, predicate-only down lookahead, projection exclusion, cancellation/error
propagation, exact strict output, and Q15's multi-statement execution. Relevant push, morsel-scan,
DuckDB, DataFusion-opening, formatting, and strict-clippy checks were also run during candidate
development before the frozen binary was created.

Final checkpoint validation is consolidated under
`/private/tmp/duckdb-frontier-final-validation.8napci`. It records:

- `git diff --check`, `cargo +nightly fmt --all -- --check`, and changed-file
  `clang-format --dry-run --Werror` passes;
- `vortex-morsel-push` with `_test-harness` (`163` passed), `vortex-morsel-scan` (`14` passed),
  `vortex-duckdb` (`224` passed, `2` ignored), `vortex-io` with `object_store,tokio` (`201`
  passed), `vortex-file` with `object_store` (`162` passed across unit/integration targets), and
  `vortex-layout` (`253` passed) test passes, plus the CUDA diagnostic-ID forwarding test;
- successful `cargo check --all-targets` and `cargo clippy --all-targets -- -D warnings` for
  `vortex-morsel-push`, `vortex-morsel-scan`, `vortex-duckdb`, `vortex-io`, `vortex-file`,
  `vortex-layout`, and `duckdb-bench` with its `unstable_encodings` feature.

The transcript retains three resolved build-environment incidents: the first morsel-scan command
hit the repository-documented `sccache: Operation not permitted` condition and passed unchanged
with `RUSTC_WRAPPER=`; the first layout build exhausted disk space and passed unchanged after only
the generated `target/debug/incremental` cache was removed; and one check reused stale
`vortex-metrics` target metadata and passed unchanged after `cargo clean -p vortex-metrics`. The
only source correction made by final validation moved an existing layout-cache test module below
production items to satisfy `clippy::items-after-test-module`; formatting and the affected clippy
gate passed afterward.

The frozen binary artifact records the isolated `release_debug` build command and two exact/plan Q6
smoke arms. The full-suite artifact records 44 plan gates and 352 fresh query processes. Treat those
reports as evidence for their recorded source patch and binary only: final validation did not rebuild
the benchmark binary or rerun the V1/W8 timing campaign. A production release still requires a build
and complete performance/correctness campaign from the exact candidate commit. Never cite a
diagnostic/profile run as timing evidence.

## Retained and removed experiments

Retain in the next candidate:

- DuckDB scheduler thread propagation;
- strict deterministic result and EXPLAIN output, still opt-in;
- current predicate-only same-range right lookahead (`2`), down `0`, refill `32`;
- W8's external-only adjacent-pair mechanism and occupancy gate as an opt-in experiment;
- the DataFusion-only weak raw-source sharing and bounded-gap/coalescing code only as separately
  identified experimental work, not as DuckDB W8 evidence;
- only low-cardinality diagnostics needed to close memory/scheduling attribution.

Parked in the dirty worktree, but not accepted:

- persistent local ObjectStore `File` payload promotion for DataFusion;
- its associated lifetime/cancellation API surface until it has exact mechanism and same-binary
  performance evidence.

Remove or keep out of the production patch:

- SourceRegistered / prepared projection and its shared/cache production machinery;
- unconditional B2/W1;
- the ineffective `!additional_filters.is_empty()` required-filter gate;
- combined mixed-priority `IoDemand::Start` and projection preparation experiments;
- intrusive per-range/per-child tracing from performance builds;
- obsolete manual thread-hint environment controls;
- temporary diagnostics whose only purpose was a completed forensic run.

## Open risks and unproven scope

- **Resource failure:** final W8 RSS and footprint fail the frozen acceptance bounds. This is the
  immediate blocker to default enablement.
- **LIMIT:** W8 is disabled only when the `MorselScan` carries a limit, but DuckDB currently does
  not propagate SQL LIMIT into `MorselScanBuilder`. Therefore the present check does not prove that
  a SQL LIMIT query is excluded. Limit propagation, cancellation, and output ordering need a
  dedicated exact suite before eligibility expands or W8 is enabled by default.
- **Experimental naming:** `VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE` is misleading and temporary; W8 is
  plan/data driven, not Q6-specific. Productionize configuration without silently enabling it.
- **Default off:** W8 remains opt-in. No result here authorizes changing the default.
- **Predicate-only baseline:** the default right-`2` predicate policy needs its own final same-binary
  production comparison against right `0`; current W8 results include it in the PF arm.
- **ConversionCache:** lifetime, entry count, byte weight, and contribution to Q8/Q10 memory are
  unmeasured.
- **HOT/local only:** no verified cold-cache, HTTP, S3, remote-latency, or remote-amplification result
  exists.
- **Memory law:** evidence is concurrency-sensitive but does not yet prove a strict
  data-independent upper bound for the whole process.
- **Scheduler cost:** elevated system time, cycles, and context switches remain unattributed.
- **Activation provenance:** the complete activation table is from the previous diagnostic binary;
  only Q6 was rechecked after the thread bridge.
- **Output format:** strict text canonicalization is sufficient for this TPC-H set, not yet a
  universal scalar-bit artifact format.
- **Worktree provenance:** the accepted result came from a frozen dirty patch, not a clean commit.

## Prioritized next experiments

1. **Close memory attribution before optimizing further.** Add fixed-cardinality, diagnostics-only
   high-water counters for live exporter rows/bytes, raw/decoded cells, `ConversionCache`, active
   local states, blocking queue, and active reads. Verify diagnostics-off structure is unchanged.
   Run Q8/Q10/Q17/Q20 at SF10 P1/P2/P4/P8/P14 and abort W8 productionization if retained bytes grow
   with completed work rather than the active window.
2. **Isolate the predicate-only baseline.** Same frozen binary and HOT protocol, PF right `0` versus
   right `2`, W8 disabled, first Q6/Q12/Q20 then all Q1-Q22. Reject if global gain is absent or any
   credible regression/resource breach appears.
3. **Retest W8 after memory fixes.** Repeat B1/W8, then full V1/W8 with a clean committed binary.
   Require exact output/plans, the existing time gates, and the existing per-query/child RSS gates.
4. **Characterize scheduler overhead.** Use low-volume counters for blocking queue/service time,
   exporter backpressure, active DuckDB tasks, and read-call/range totals. Profile only after a
   diagnostic-off timing result identifies a target; profiles remain non-timing evidence.
5. **Expand scope last.** Add LIMIT correctness/cancellation, then verified cold local and remote
   HTTP/S3 protocols. Do not tune gaps, morsel size, or right/down breadth until each change has a
   same-binary mechanism result and bounded-memory argument.

Abort a candidate immediately on exact output/plan mismatch, a noncontiguous/duplicate/missing
range, failure to clean up cancelled work, data-size-dependent retained live state, any credible
greater-than-5% query regression, or the existing RSS hard limit.

## Artifact index

| Purpose | Artifact | Manifest SHA-256 | `SHA256SUMS` SHA-256 |
|---|---|---|---|
| Canonical V1/W8 full 22 | `/private/tmp/duckdb-thread-bridge-v1-w8-full22.ahG9bN` | `3dd2690caf74d47efe1e66d180a66549e89a513fbc2a717fdcdf7b798a4951db` | `d35ffdcd054db36a624a85eb4508785979c9a9934153f196f1583e74c3fb2463` |
| B1/W8 full 22 + activation | `/private/tmp/duckdb-w8-full22.0wO1Zw` | `d1142ae7cc79c6e26d13705afebd77a250372f76fe78617bfc8c67f1ecc9bab3` | `2299ed2a24f192e0a6bfa1646024021be0a1d41e0f62aeafa6606db4dcbc8f94` |
| Post-thread-bridge Q6 | `/private/tmp/duckdb-thread-bridge-q6-ab.Dm5bhd` | `51cda179e7a78c9dcdec2fc15c2f789606e2a331dba0f30d5dda51aa9bddf619` | `ed26e634f9bb1ead421f26b234c385f297de22cd0c5ce6b669cfb5b1e5c08c27` |
| Rejected SourceRegistered | `/private/tmp/duckdb-q6-source-reg-ab.v5B3rR` | `d94d8a72658e8f0a930deb7260254a755c36647f187306662c4aa96a0af53498` | `949f39cefdfc8c8e24f3c3c4edd394fd025acd4dd34cc7c6fbbfb97deccd1b7c` |
| Rejected unconditional B2 | `/private/tmp/duckdb-tpch22-bundle2-perf.u6Q4Ug` | `73559bcea0dda281412fc1d2082ec062c75bccb46bbdf03082a60a0db85e372b` | `64db18af1b937fe504cd8c84984968679074b854a422501db904ce2a841ea8a4` |
| Rejected filter gate | `/private/tmp/duckdb-required-filter-ab.x8cdpa` | `cb80a196a61eaf1b94d453c3729424dcda7955d31a3ce10413e14a33e29fdad9` | `d5a710d356ddce5da9f89e3c70c30ee6ff04d5a31e74e05a0d98cd0f939f5112` |
| B1/W8 focused memory | `/private/tmp/duckdb-b1-w8-memory.NPeKjU` | `49834b11b01d632f02ce931a9d858c6ceee3df3f73af68fe4c5cec8fe3ba42fa` | `e5c823639b2936b0fee8c184eb5bce665585ca21e9743ac1d2404033b7c382f5` |
| W8 memory scaling | `/private/tmp/duckdb-w8-memory-scaling.IqJzs2` | `f75990fb4a9e4b5392b4ec75f8cb8a406ba218e2ab2010ec39f437fbf9c8b76a` | `fc6de5b18b02a6adc5ef73a88efafcdcfe1f84c80e792378d1cc7f5fbc90de95` |
| SourceRegistered profiles | `/private/tmp/duckdb-q6-source-registered-samply-20260913.9KAQ6V` | No manifest; selected-file `HASHES.md` SHA: `1e26efd8ce917c09bcd17c894de5ed4cf56b5ab7a187f7e5164f601556a000c9` | No `SHA256SUMS` or comprehensive tree seal; `REPORT.md` SHA: `90eaca36c805cad2fa8f4aad2c4edecdf6ced77b0cc29678a2d12700f61c40df` |

The `/private/tmp` locations are host-local and not durable repository assets. Before cleanup, copy
the canonical root and every retained supporting artifact to durable storage without changing the
files, then record new location hashes here.

## Resume checklist

- [ ] Verify the canonical binary, manifest, and `SHA256SUMS` hashes before reading conclusions.
- [ ] Confirm branch/HEAD and inspect the complete dirty diff; do not assume the worktree equals the
  frozen patch after this documentation update.
- [ ] Separate retained production changes from rejected and diagnostic WIP before committing.
- [ ] Preserve V1, push, and push-frontier backend selection and identical default fields/files.
- [ ] Keep W8 opt-in, external-only, filtered, contiguous-pair, and W8-gated; do not claim SQL LIMIT
  exclusion until DuckDB propagates the limit into `MorselScanBuilder`.
- [ ] Run exact output and raw/normalized plan gates before any timing.
- [ ] Use the same frozen binary, fresh-process HOT prewarm, balanced AB/BA order, and no concurrent
  build/profiler for both arms.
- [ ] Never mix diagnostic/profile children into performance summaries.
- [ ] Close the RSS/footprint failure before proposing default enablement.
- [ ] Record a clean commit, binary, data, SQL, command, host, and full hash manifest for the next
  campaign.

## Historical evidence

The following material explains design choices but is not the current DuckDB acceptance result.

### Direct morsel harness

The earlier in-process SF10 direct harness reported approximately `1.482x` HOT, `1.587x` cold, and
`2.889x` with injected 2-ms latency for its strongest frontier policy. That harness used a different
engine boundary, work ownership, timing layer, and policy (six resident morsels per worker, 64 down
frontiers per worker, adaptive right traversal, 256-range refill). It established that frontier
visibility can hide I/O latency; it did not prove production DuckDB or DataFusion speed.

### DataFusion public matrices

[PUSH_FRONTIER_BENCHMARK_CHECKLIST.md](PUSH_FRONTIER_BENCHMARK_CHECKLIST.md) retains the complete
DataFusion SF1 TPC-H, ClickBench, FineWeb, and TPC-DS inventory/correctness/timing matrices. Those
runs used the then-current DataFusion integration, generally four target partitions/workers for
performance and one for exactness, and historical `0/0/32` frontier defaults. They should not be
used to override the DuckDB SF10 result or describe the current policy.

Historical DataFusion Q6/Q22 diagnostics found fragmented small reads and overlap without measured
frontier lock contention. Source pooling, persistent local-file descriptors, aggregate gap budgets,
and shared-completed-segment caching were explored; several were rejected or left unmerged because
they amplified bytes, did not improve the target, or expanded API/lifetime complexity. The durable
lesson is to prove visibility, physical-byte union, cancellation, and bounded lifetime before
tuning coalescing.

For detailed historical scheduler implementation notes, see
[IO_FRONTIER_IMPLEMENTATION.md](IO_FRONTIER_IMPLEMENTATION.md) and
[IO_FRONTIERS.md](IO_FRONTIERS.md). Their numeric results and default descriptions are historical
unless repeated in the canonical artifact above.
