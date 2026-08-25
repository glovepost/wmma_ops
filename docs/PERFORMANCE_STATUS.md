# gfx1151 WMMA performance status

This is the current performance ledger and optimization plan for the repository.
It was refreshed on 2026-08-25 with the ordinary-layout standalone harness,
the block/K16-prepacked research harness, the Paperclip literature pass, and
the current assembly-search tooling. In-progress extension-kernel changes in
the shared worktree were deliberately not included. The older development
notes remain useful as a lab notebook, but their claims are not automatically
current.

## What the repository has actually measured

The ordinary-layout contract has a validated peak of **41.322 TFLOPS** (3.326
ms) for a 4096 x 4096 x 4096 GEMM with FP16 inputs and FP32 accumulation/output.
It uses the pinned standalone harness in `tools/`, deterministic random inputs,
and a full rocBLAS FP32 reference over all 16,777,216 outputs. Its five-process
ordinary-layout audit had a 40.900 TFLOPS median and a 40.772-41.104 range. The
separate block/K16-prepacked FP16-output contract is the current sustained
research leader at **50.074 TFLOPS** and has cleared its five-process
50-TFLOPS promotion gate.

The previous documented record was **21.6 TFLOPS**. It came from the historical
ROCm 7.9/7.10-preview environment and used three warm-ups and 20 timed
iterations. The raw output, exact clocks, temperatures, compiler build, and
run-to-run distribution were not committed, so it remains a historical
reference rather than the reproducible current result.

There are two other numbers in the old documentation that must not be compared
directly with the record:

- 20.61 TFLOPS for `matmul_zerocopy` used a 4096 x 4096 x 2048 shape.
- 21.9 TFLOPS for the standard kernel appeared without a raw result or complete
  run description. Treat it as an observation, not a replacement record.

### 2026-08-23 upstream FP16-output comparison

The current upstream
[`adelj88/rocm_wmma_gemm`](https://github.com/adelj88/rocm_wmma_gemm) benchmark
was also reproduced on the shared gfx1151 host. Commit
`281b5dfd7fbff9cea80753bc55274a54f4a7c53a` was built for `gfx1151` with ROCm
7.14 and `/opt/rocm/bin/amdclang++`. Each timing run used the upstream default
Google Benchmark protocol, without profiler counters:

```bash
./benchmark/bench_half_half --shapes 4096
```

For the published 4096 x 4096 x 4096 layout with A column-major, B row-major,
and C row-major, three fresh benchmark processes reported:

| Process | Average TFLOPS | Reported time | Minimum time | Maximum time |
|---:|---:|---:|---:|---:|
| 1 | 45.8997 | 2.99 ms | 2.94046 ms | 3.06502 ms |
| 2 | 46.0820 | 2.98 ms | 2.94812 ms | 3.02114 ms |
| 3 | 46.2115 | 2.97 ms | 2.92912 ms | 3.09592 ms |

The median is **46.082 TFLOPS**, with a 45.900-46.212 TFLOPS range. That is
7.37% above the upstream
[published 42.92 TFLOPS result](https://github.com/adelj88/rocm_wmma_gemm/blob/main/docs/gfx1151_square.md)
for the same shape, layout, and `bench_half_half` contract. The entire
eight-layout benchmark was run in each process so the selected row retained the
published invocation's ordering and warm state.

All 208 upstream same-precision correctness tests passed across FP16, BF16, and
all eight layout combinations. Those tests cover several shapes through
1024 x 256 x 256, but they do not compare the full 4096-cubed result against a
reference. The benchmark itself times the kernel without checking its output.
Accordingly, 46.082 TFLOPS is a reproduced external FP16-input/FP16-output
comparison, not a promoted `wmma-ops` record and not an apples-to-apples
replacement for the FP16-input/FP32-accumulation/output result above.

### 2026-08-22 ROCm 7.14 sweep

The active gfx1151 sweep has superseded 21.6 TFLOPS as an engineering
baseline and crossed 41 TFLOPS in validated processes, but has not completed
the all-five-process promotion gate. All
entries below use the record shape, FP16 A/B, FP32 accumulation/output, and the
normalized-maximum correctness threshold below 1%. Allocation and API timing
contracts differ by harness and are called out separately. Timing and profiler
counters are collected in separate processes.

| Candidate | Reported TFLOPS | Result |
|---|---:|---|
| Pinned rocWMMA `1,8,7,2,2,1,8,128`, CU mode | **41.322 peak** | 3.326 ms; full-output validated, 200 warm-ups and five 30-launch blocks |
| Same candidate, fresh-process promotion audit | **40.900 overall** | 40.772-41.104; 2/5 process medians above 41 |
| `matmul_opt` | 24.762 | Local unmerged build, correct in five fresh processes; binding is in source base |
| `matmul_opt_interleaved` | 29-30.5 | Local unmerged extension experiment; varies with shared-box state |
| `matmul_opt_k32` | **30.870** | Local unmerged extension experiment; 30.804-31.180 block range |
| AMD Tensile source 91, exact 4096 specialization | **37.802** | Correct over 200 warm-ups and 100 timed iterations; external, not promotion-tested |
| Source 91 with favorable B allocation phase | 38.468 | Correct 10-iteration screen; allocation-dependent, not a record |
| Source 91 control in the post-profile screen | 38.185 | Correct 20-iteration control; no promotion |
| Independent rocWMMA FP32-output tuner | **38.07** | Best of 1,000 validated row/row/row configs; 3.610 ms at evaluation 826 |
| Source 91, B wave separation disabled | 37.760 | Correct short screen; rejected |
| Source 91, alternate B padding | 37.379 | Correct short screen; rejected |
| Source 91, A-major LDS plus padding | 36.868 | Best correct A-major short screen; rejected |
| Source 91, WorkGroupMapping 2 | 34.616 | Correct; mappings 4 and 8 were slower |
| Source 91, StaggerU 8 | 34.213 | Correct; stagger 16/32/64 were slower |
| 128x64 K32, hipEngine-style lower live state | 23.873 | Rejected |
| Source 91, two LDS buffers | 23.596 | Correct; rejected |
| raw-builtin 128x128 K32 | 20.862 | Rejected |
| no-pad/streamed-fragment K32 | 17.275 | Rejected |
| CU-mode K32 | 23.553 | Rejected; WGP mode restored |
| single-wave 32x64 | 9.257 | Rejected |
| persistent simple `[N,K]` prepack | 6.872 | Separate inference diagnostic; rejected |

### Validated 41 TFLOPS crossing

The winning standalone specialization is
`1,8,7,2,2,1,8,128`: one warp in M, eight in N, a 7 x 2 WMMA tile per wave,
two K slices, one LDS buffer, swizzle 8, and 128-bit global loads. That gives a
112 x 256 x 32 workgroup tile with 256 threads. The emitted kernel uses CU mode,
188 VGPRs, 30 SGPRs, 25,344 bytes of LDS, and no scratch spills. It is compiled
from `adelj88/rocm_wmma_gemm` commit
`281b5dfd7fbff9cea80753bc55274a54f4a7c53a` with ROCm 7.14 and
`-mllvm -amdgpu-unroll-threshold-local=700`.

The best validated search sample used one combined A/B allocation with B
placed 2,048 half elements after the end of A:

```text
median_ms=3.326064 tflops=41.321799 min_ms=3.324035 max_ms=3.375994
validation finite=yes elements=16777216 max_abs_error=0.000408173
reference_max_abs=30.393165588 normalized_max_error=0.000013430
rms_error=0.000067966
```

The stronger fresh-process audit used 200 warm-ups and five blocks of 100
launches in each process. Its process medians were 41.103701, 40.881723,
40.771988, 41.071171, and 40.900087 TFLOPS. This proves the target is reachable
with correct FP32 output, but not yet that the kernel stays above it across
normal process variance. Two clean-rebuild audits after the extended tuning
session validated all ten outputs but measured only 40.244-40.939 TFLOPS
(0/10 above 41), consistent with the longer package-power window observed
below. The raw logs are:

- `/root/wmma-results/rocwmma-threshold-sweep-20260823T014614Z.log`
- `/root/wmma-results/rocwmma-memory-phase-20260823T020903Z.log`
- `/root/wmma-results/rocwmma-long-warmup-20260823T021444Z.log`
- `/root/wmma-results/rocwmma-final-audit-20260823T022231Z.log`
- `/root/wmma-results/rocwmma-final-audit-20260823T022551Z.log`

Reproduce the candidate from the repository root:

```bash
git clone https://github.com/adelj88/rocm_wmma_gemm.git /tmp/rocm_wmma_gemm
git -C /tmp/rocm_wmma_gemm checkout \
  281b5dfd7fbff9cea80753bc55274a54f4a7c53a
tools/build_rocwmma_record.sh /tmp/rocm_wmma_gemm build/rocwmma_record
build/rocwmma_record 200 100 0 2048 0 combined-ab
```

Raw results are retained on the gfx1151 host under `/root/wmma-results/`.
Those absolute paths are machine-local and are not repository artifacts; the
important validation sample and fresh-process distribution are reproduced in
this document, while the harness allows an independent rerun.
These figures are not mixed with profiler durations. The leader's standalone
profile measured about 21.5% of wave cycles waiting at barriers and 11.2%
waiting on LDS instructions, with about 59.5% reported occupancy. That makes
barrier/LDS scheduling the measured optimization target.

The historical “rocBLAS 41 TFLOPS” number is not the same numerical contract:
the old comparison used `torch.mm` with FP16 output. A direct ROCm 7.14
`rocblas_gemm_ex` probe on this host measured 34.388 TFLOPS for
FP16×FP16→FP16, but only 7.033 TFLOPS for FP16×FP16→FP32; the latter selected a
generic 64x32x8 kernel without matrix instructions. The 41.322 TFLOPS
standalone peak is about 5.9x that fair rocBLAS FP32-output path and also
crosses the harder historical 41 TFLOPS observation.

The tuned FP16-output rocBLAS trace selected a gfx1151-specific
128x128x32 macro-tile with four wave32s, a 4x4 WMMA tile per wave, one LDS
buffer, 17,408 bytes of LDS, and 256 VGPRs. Adapting AMD's official Strix Halo
Tensile logic from FP16 output to FP32 output preserved that instruction
schedule and produced correct results, but a complete 100-solution screen has
not crossed 41 TFLOPS for contiguous 4096 matrices. The current external leader
is an exact-size depth-32 schedule at 37.802 TFLOPS. It uses four wave32s, a
4x4 WMMA tile per wave, 17.5 KiB of LDS after A-side padding, 256 VGPRs, and no
scratch spills. A favorable B allocation phase reached 38.468 TFLOPS in a
short screen, but allocation sensitivity is not a kernel record. A clean-room
source experiment with
the same four-wave output geometry was correct but reached only 19.079 TFLOPS
when both operands used double-buffered LDS. The later one-buffer and scheduled
global-prefetch variants confirmed that geometry alone is not the optimization.

The exact-size sweep also rejected several plausible shortcuts. Doubling the
workgroup to eight waves while shrinking each wave's tile reached 34.9 TFLOPS;
an eight-wave 128x256 macro-tile that retained the 64x64 per-wave tile reached
30.6 TFLOPS. Persistent workgroups peaked at 36.2 TFLOPS. Prefetch depth two,
beta-zero specialization, alternate scheduler issue rates, hipBLASLt, and
larger A/B padding phases all stayed below the exact source-91 result.
Store-remap and direct-to-VGPR variants are not timing results: the pinned
Tensile generator rejects those WMMA/FP32 combinations before assembly. The
same is true of source-91 `PrefetchLocalRead=1`: the plain, double-buffered,
and A-major combinations all failed kernel generation, so none has a TFLOPS
number.

The first WorkGroupMapping/StaggerU batch also produced no code objects, but
that was a generator-input bug rather than a kernel rejection: source 91 stores
one shared `VectorWidth`, while those rederived paths index `VectorWidthA` and
`VectorWidthB` directly. Supplying the equivalent per-tensor aliases generated
all eight kernels. Their correct measurements are in the table above; do not
cite the earlier batch failure as a performance result.

The independent rocWMMA search has now evaluated 1,000 unique configurations
(31.6% of its 3,162-config row/row/row space) with FP16-rounded inputs and FP32
output validation enabled. Its best configuration is
`2,4,3,3,2,1,8,128` (`warps_m`, `warps_n`, `warp_tile_m`, `warp_tile_n`,
`k_slices`, `single_buffer`, `swizzle`, `bits`) at 3.610 ms, or 38.07 TFLOPS.
That improves the earlier 300-evaluation 3.620 ms result but does not meet the
3.352 ms / 41 TFLOPS gate. The raw log is
`/root/wmma-results/f32-tuner-20260823-20260822T223427Z.log` on the gfx1151
host.

A focused basin search around the tuner output found the substantially faster
`1,8,7,2,2,1,8,128` schedule. The subsequent robustness campaign rejected the
following as durable improvements:

- LLVM local-unroll thresholds from 700 through 800 and scheduler flag
  combinations changed individual processes but did not lift the floor.
- `amdgpu_waves_per_eu` bounds produced peaks up to 41.288 TFLOPS, but the best
  finalist still fell to 40.885 TFLOPS.
- 128-row tiles that divide 4,096 exactly reached only 29.0-37.5 TFLOPS.
- A separate 64-row tail launch reached about 39 TFLOPS; duplicating an edge
  path in one kernel reached about 36.6 TFLOPS.
- Padded physical M=4,144 and unchecked output stores reached only
  40.5-40.8 TFLOPS.
- Workgroup mappings, split-tail variants, A/B/C phase sweeps, and longer
  1,000-launch warm-up did not remove fresh-process variance. The longer
  warm-up instead exposed package-power decay down to 40.238 TFLOPS.

The AMD RDNA3.5 system-optimization audit found no missing host throughput
setting. The host kernel is newer than AMD's listed minimum, the TTM page limit
and rocminfo pool expose about 124 GiB for the APU, sysfs reports AMD's example
512 MiB VRAM carve-out, and the GPU is held at the high performance level. AMD
documents the TTM limit and BIOS UMA carve-out as capacity controls; increasing
either is not a GEMM-throughput optimization.

The 59.4 TFLOPS nominal ceiling is consistent with the AMD Matrix Instruction
Calculator: `v_wmma_f32_16x16x16_f16` performs 8192 FLOPs in 32 execution
cycles and is rated at 1024 FLOPs/WGP/cycle on gfx1151. With 20 WGPs at a
2.9 GHz peak clock, that gives 59.4 TFLOPS. It is a clock-dependent nominal
ceiling, not a measured sustained ceiling.

The integer sibling `v_wmma_i32_16x16x16_iu4` performs 8192 integer operations
in 16 execution cycles and is rated at 2048 operations/WGP/cycle, giving a
118.8-TOPS nominal ceiling at the same clock.  A standalone signed-IU4 issue
test reached 110.229 TOPS (92.8% of that ceiling) with 16 independent chains;
four, eight, and twelve chains reached 108.380, 109.408, and 109.718 TOPS.
Even one chain reached 103.040 TOPS because resident waves hide most of the
dependency latency.  A separate nonuniform 16x16 product validates all 256
outputs against the ISA's replicated-operand and split-row result mapping.
These are integer TOPS, not TFLOPS, and do not change the FP16 record.

A complete prepacked linear-W4A4 GEMM retains 85.907 INT4 TOPS at 4096 cubed
(1.599854 ms) while matching all 16,777,216 INT32 outputs exactly.  Its
128x128 block has eight waves, eight independent accumulators per wave,
double-buffered conflict-free packed LDS, one barrier per K16, 89 VGPR,
22 SGPR, 4,096 bytes of LDS, and no spills.  Repeating the two wave geometries
in one exclusive pass favored four wave columns: 85.063 and 85.907 TOPS versus
84.992 and 84.182 for two wave columns.  A padded 12-byte LDS row reached only
83.416 TOPS, and a 256x128 block rose to 159 VGPR and fell to 78.367 TOPS.

The paired-K IU4 experiment keeps two K16 slices in four rotating LDS slots,
stages the following pair while the current pair computes, and publishes once
per pair. With `-DIU4_PAIR_K=1` it compiled at 103 VGPR, 18 SGPR, 8,192 bytes
of LDS, and zero spills. Five interleaved fresh pairs measured
90.527/90.208/89.381/90.599/90.129 INT4 TOPS (90.169 average, 89.381 floor),
with zero mismatches across all 16,777,216 INT32 outputs. The one-slice control
measured 84.961/84.928/84.942/84.829/85.138 (84.960 average). This is a
substantial integer-kernel improvement, not a promotion of the FP16 TFLOPS
record; translating it to FP16 requires a different LDS/register budget and
must be measured independently.

This result demonstrates that a full data-moving IU4 kernel can exceed the
50-operations/s target, but it remains a distinct integer contract.  It is not
eligible for the FP16 TFLOPS table or record gate.

The current FP16-output, block/K16-prepacked research leader averages **50.074
TFLOPS** at 4096 cubed across five fresh alternating-order candidate/control
pairs. Its medians are 50.015/50.136/50.146/50.026/50.047 TFLOPS, with a
50.015-TFLOPS floor and 2.744730-ms average median time. Paired selected-image
controls average 49.019 TFLOPS. It uses a 256x128 block, eight waves, p8 A/B
LDS rows, 120 VGPR, 22 SGPR, 18 KiB LDS, and no spills. The result passed a
full rocBLAS reference check in every process, but it is a persistent-input
contract: packing is outside the timed region. The prior qualified leader
averaged 49.143 TFLOPS. The sustained 50-TFLOPS FP16 gate is now met.

A fresh five-process recheck after the shared-box load-settle experiment
measured **48.939/48.912/48.847/48.633/48.504 TFLOPS** (48.767 average,
48.504 floor). Every process passed the exact numerical tuple. This confirms
that a warmer package state or loader ordering does not close the gap; the
promotion gate remains five fresh processes above 50.

A Paperclip-guided WMMA issue-order screen then permuted only the three
fully-ready four-row fragments in the delta-2 hot loop. The progressive first
fragment was left untouched, and row 0 remained first in the final fragment
because the following global refill overwrites its A registers. Every form
assembled at 120 VGPR, 22 SGPR, 18 KiB LDS, zero spills, and reproduced the
full reference tuple. In a low-package-state exploratory bracket, controls
reached 44.954/44.808 TFLOPS; row orders 0132/0213/0231/0312/0321 reached
44.853/45.231/44.855/45.348/45.274. The 0312 signal is +1.04% over that
bracket's control midpoint, but these absolute values are not comparable to
the 49.035 qualification and the run was interrupted for an Ember release.
The isolated group-1/group-2/group-3 follow-up is prepared in
`build-wmma-row-order-isolation.sh` and remains unqualified until the shared
GPU is available.

That isolation bracket subsequently ran with the full correctness gate on all
eight forms. Opening and closing controls were 49.295/49.176 TFLOPS. The
group-specific results were g1 49.285, g2 49.329, g3 49.078, g1+g2 49.006,
g1+g3 49.091, and g2+g3 49.095 TFLOPS. The all-group 0312 form reached 49.401
TFLOPS, but the control drift and the lack of a sustained fresh-process screen
keep it below promotion. Every process reproduced normalized maximum error
0.018779343, RMS 0.035428338, and cosine 0.999977929. Retain all-group 0312
as the next short-screen candidate; do not replace the delta-2 leader yet.

The required fresh-process comparison then closed that candidate. With 20
warmups and five 100-iteration timing blocks per process, the base medians were
49.173/49.162/48.910/48.798/48.694 TFLOPS. The all-group 0312 medians were
49.066/48.809/48.515/48.834/48.830 TFLOPS. Every process was exact, but the
candidate lost to its immediately preceding control in all five pairs. Close
row-order permutations as noise and keep the delta-2 allocation unchanged.

The latest occupancy-preserving barrier experiments did not close that gap.
A hand-scheduled compact interleaved ping-pong kernel retained 118 VGPR, two
blocks/16 waves, 30 KiB LDS, and one barrier per K16, but reached only 46.024
TFLOPS in its best row-pitch placement.  A periodic compact-A/p8-B form fit
exactly 32 KiB and reached 44.947 TFLOPS.  A 512x128, 16-wave block was exact
at 124 VGPR and 30 KiB, but gfx1151 still admitted only one block/16 waves and
it reached 35.850 TFLOPS.  Lowering CPU energy preference while keeping the
GPU at 2.9 GHz changed the p8 leader by only about 0.35%, so package policy is
not the missing 2.8%.

Later producer/consumer, prefetch, and epilogue experiments also preserved the
48.614-TFLOPS leader.  Pair-local LDS flag handoff was exact but reached 44.723
TFLOPS; front-loading a complete refill reached 45.714--46.100 TFLOPS; and a
corrected DPP vector epilogue reached 47.335 TFLOPS.  The epilogue correction
matters independently of speed: RDNA 3.5 DPP `bank_mask` selects four-lane
groups, so it cannot directly select lane-id bits 0 and 1.  The earlier output
from that mistaken assumption was invalid and is not a timing result.

Replacing flat-address recurrence with scalar-offset MUBUF is the one small
positive signal.  Four longer interleaved runs averaged 47.786 TFLOPS versus
47.613 for the controls (+0.36%) with unchanged 118-VGPR, two-block occupancy.
Clause removal, SALU reordering, and advancing only the B refill did not improve
it.  This is retained as a code-generation building block, not promoted as a
new record: it remains within normal run-to-run variation and below 50 TFLOPS.

A Paperclip-guided four-wave revisit found one substantial but insufficient
architecture gain. Preventing LLVM from hoisting all four B fragments lowered
the 128x128 kernel from 153 to 129 VGPR, raised runtime residency from four
blocks/16 waves to five blocks/20 waves, and improved throughput from 41.419 to
45.694 TFLOPS (+10.3%). Splitting the A/B refill live ranges and using one
MUBUF vector offset reached 120 VGPR with zero spills, but the 12-KiB LDS tile
already limits residency to five blocks. The 127/121/120-VGPR forms therefore
remained at 20 waves and reached only 45.011/44.630/44.839 TFLOPS.

A follow-up crossed both resource thresholds. Streamed-B p8p2/p4p4/p2p0/p0p0
layouts used 10.5/10/8.5/8 KiB LDS and reached
40.608/26.696/37.495/45.445 TFLOPS. Combining split refill lowered them to
124/122/120/119 VGPR. The first two then reported six blocks/24 waves but
reached only 40.854/26.495 TFLOPS; the last two still reported five blocks/20
waves and reached 35.426/43.796. All were exact. The bracketed p8 stream
controls reached 45.827/45.698 and retained-kernel controls reached
48.001/48.234 TFLOPS.

The negative result is conclusive for this family: added nominal occupancy
does not rescue a poor LDS bank phase, while split refill loses useful
B-load/WMMA overlap. Static `.vgpr_count` also proved insufficient for
occupancy inference because `.amdhsa_next_free_vgpr` stayed 169 and the runtime
query was non-monotonic. At this stage the 129-VGPR p8 streamed form remained
the best four-wave candidate and the overall FP16 leader remained 48.614
TFLOPS; the later register-phase result supersedes the latter number.

Single-operand ping-pong also failed to advance the leader. Double-buffering
only A preserves two blocks/16 waves at 127 VGPR and 30 KiB LDS; split,
grouped, and late-`vmcnt(1)` schedules reached 44.569--44.921 TFLOPS. The
B-only sibling used 129 VGPR and 24 KiB LDS and reached 45.178/45.202 TFLOPS.
All were exact, while their bracketed p8 controls reached 47.809--48.082.
These forms move safe LDS stores into the WMMA cluster but retain both
workgroup barriers, so the added LDS contention and wait threshold cost more
than the shorter serial handoff saves. The opt-in implementation leaves the
default device code instruction-identical.

The remaining p8 paired-LDS encoding gap is also closed. Two loop-invariant
512-byte-shifted bases make the fourth `ds_load_2addr_b64` fragment fit its
eight-bit offsets. The kernel is exact at 121 VGPR and 18 KiB LDS and reports
three blocks/24 waves, but paired reads/stores reached only 44.468/44.428
TFLOPS versus 47.914/47.991 controls. Restoring native b128 stores did not
rescue the paired reads: it reached 43.442/42.041 versus same-pass
44.932/45.055 controls. The native p8 `ds_load_b128` schedule remains best.

An assembly-qualified progressive refill handoff is the next small positive
signal. It moves the refill waits after the overwrite barrier and commits A0,
A1, and B at `vmcnt(2)`, `vmcnt(1)`, and `vmcnt(0)` respectively. The kernel
remains exact at 118 VGPR, 22 SGPR, 18 KiB LDS, zero spills, and two reported
blocks/16 waves. Four longer interleaved runs averaged 47.948 TFLOPS versus
47.790 for their immediately preceding controls (+0.33%). Like scalar-offset
MUBUF, this is retained as a composable scheduling improvement rather than a
record. Combining the two exact 118-VGPR transforms averaged 48.173 TFLOPS in
a longer interleaved screen, versus 47.812 for progressive-only and 47.756 for
the controls: +0.75% and +0.87%, respectively. The combined schedule becomes
the next research base, but the absolute pass remained below the historical
48.614-TFLOPS leader and the sustained 50-TFLOPS gate remains open.

A one-fragment B lookahead is also closed. It pipelines each future B LDS load
across the current four-WMMA group at 126--132 VGPR. All corrected placements
were exact, but reached only 45.203--45.999 TFLOPS against 48.352/48.276
combined-base controls. Alternate B base placement changed throughput by
1.3--1.6% within equal reported-occupancy classes, proving register assignment
matters, but the best form still lost 4.8%. The 126/128-VGPR forms reported
three blocks/24 waves; those extra waves did not repay deeper LDS queuing and
the alternate WMMA operand bank.

A semantics-preserving register-boundary shift produced the new leader. It
holds accumulators at v1--v64 and shifts every address/fragment/refill VGPR
together, changing only the physical WMMA operand-to-accumulator phase. Odd
deltas violate the gfx1151 even/odd destination rule for an existing dual-VALU
instruction, so instruction-identical deltas 2/4/6 were tested. Delta 2 reached
49.439 TFLOPS in the short bracket at 120 VGPR and the same reported 16 waves;
delta 4/6 reported 24 waves but fell to 45.049/46.025 TFLOPS.

Four longer delta-2 runs averaged 49.323 TFLOPS versus 48.192 immediately
preceding controls (+2.35%). The five-process 100-iteration screen then
averaged 49.035 TFLOPS against 48.103/47.941 bracketing controls, +2.11% over
their average and +0.87% over the former 48.614-TFLOPS leader. This promotes
delta 2 as the research base and leaves a 1.97% sustained gap to 50 TFLOPS.

Moving the same two-register gap to other safe live-range boundaries did not
improve it. In a lower package state, opening/closing v65 controls reached
45.510/45.302 TFLOPS. Boundaries v17/v33/v49/v57/v66/v68/v69/v70 reached
44.430/44.461/44.464/44.848/45.150/44.420/44.513/44.607, all exact. Treat this
only as a same-pass relative screen: v66 was closest at -0.56%, while all other
cuts lost 1.2--2.2%. The promoted phase group starts at v65.

Finer fixed-resource placement did not add another gain. Exchanging the hot B
bank with each A bank and exchanging epilogue-safe accumulator pairs at
v17/v33/v49 all remained exact at 120 VGPR. Initial 49.767/49.785-TFLOPS
signals reversed in the composition bracket: 49.863/49.827 controls bracketed
49.598 for B/A0, 49.567 for v17/v49, and 49.552/49.886 for their combination.
Retain the ordinary delta-2 allocation; the pair permutations are noise at
this measurement resolution.

An issue-order screen then permuted the ten independent initial `ds_load_b128`
operations in the delta-2 hot loop while leaving every destination, address,
WMMA order, and wait threshold unchanged. B-first and B-middle remained exact
but reached only 49.152 and 49.072 TFLOPS in a five-block short bracket versus
49.252 for the same-pass control. An alternating A-stripe order failed the full
reference (normalized error 239.96), showing that the existing LGKM wait
thresholds encode a real completion-order dependency even when the source
addresses are independent. Close issue-order permutations as a negative result;
the retained schedule remains the compiler order plus the delta-2 register
phase.

The source-supported `warp_tile_m=2` geometry was also screened as a genuinely
different wave architecture. A `256x128` block used 16 waves and 91 VGPR, but
the specialized loader faulted on the first dispatch, so it is invalid rather
than a timing result. A `128x128`/eight-wave form assembled at 99 VGPR but
faulted with a GPU memory access error before validation. The prefetch-position
permutations that did run (A0/A1/B at 0/1/2, 1/0/2, and 0/2/1) retained
exactness; the default order reached 49.371 in the short bracket while the
alternatives reached 49.190/49.181 versus a 49.199 control. Treat these as
closed: the default prefetch order and eight-wave `256x128` geometry remain
the research base, and the warp-tile-2 path needs a loader-contract rewrite
before it can be evaluated fairly.

A targeted loader repair for the `128x128`/eight-wave form then removed the
out-of-bounds B-vector read and ran without a fault, but its full-output check
still failed (normalized max error 0.08731, RMS 0.44221) at 37.432 TFLOPS. The
repair is not retained: the experiment proves that this geometry needs a
complete lane-to-row B mapping, not just a bounds fix.

At 256 GB/s, the corresponding compute-to-memory ridge point is about
232 FLOP/byte (`59.4e12 / 256e9`), not 106 FLOP/byte. Both inputs should be
replaced by observed clocks and sustained bandwidth when making a measured
roofline.

Primary references:

- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [AMD GPUOpen: WMMA on RDNA 3](https://gpuopen.com/learn/wmma_on_rdna3/)
- [ROCm Compute Profiler support for gfx1151](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/reference/compatible-accelerators.html)
- [Repository WMMA reference audit](wmma_references.md)

## Source audit

### The current 128 x 64 baseline

`matmul` launches a 128 x 64 work-group tile with 256 threads (eight waves).
Each wave computes a 32 x 32 tile using four independent FP32 accumulator
fragments. The K loop advances by 16, uses double-buffered LDS with a stride of
24 half values, and has one work-group barrier per K tile.

The shape is important: compared with a 64 x 64 block, increasing M to 128
reuses each staged B tile across more output rows. Older notes call this
"increased A reuse"; it is B reuse.

### Adaptive selection

At 4096 x 4096 x 4096, `matmul_adaptive` selects the gfx1151 variant with the
same 128 x 64 geometry. The `WIDE_128x128` enum exists, but
`select_optimal_tile()` currently has no path that returns it. The adaptive
selector therefore does not test a wide tile at the record shape.

### Large-tile candidate (measured and rejected)

`matmul_opt` is bound in Python but is absent from the benchmark lists. Its
actual kernel is a 128 x 128 work-group tile with eight waves and a 2 x 4 WMMA
tile per wave (eight FP32 accumulators). The binding text and wrapper comments
that describe a 256 x 128 tile, 4 x 4 warp tiling, or a vectorized epilogue are
stale.

This was measured as a first experiment. At 4096 cubed it ran 20.52--20.64
TFLOPS, and its full-output check failed badly (`normalized_max_error=1.268672511`,
cosine `0.124986872`). Passing B transposed did not repair the contract: it ran
8.00 TFLOPS with normalized error 1.286538805 and cosine 0.125113875. The
candidate is therefore both numerically invalid and far below the prepacked
leader; the stale binding description should not be used to motivate another
screen without first replacing its fragment/load/epilogue contract.

### Historical names

`matmul_zerocopy` performs vector global loads for B, followed by eight scalar
stores into transposed LDS, and uses the same scalar direct-to-global epilogue
as the standard kernel. "Zero-copy" and "swizzled B" are historical names, not
descriptions of the current data path.

## Current plan beyond 50 TFLOPS

The ordinary-layout peak remains a separate contract, while the prepacked
FP16-output kernel now averages 50.074 TFLOPS with a 50.015 floor. The next
work is to widen that margin without weakening either numerical contract.

1. **Protect the qualified floor.** Optimize against the 50.015-TFLOPS floor
   and fresh-process average, not the 50.530 short screen. Re-run the five
   process gate after any compiler, clock, firmware, or harness change.
2. **Attack synchronization with independent work, not a wider K tile.** The
   measured K32 publication stage halved barriers per K16 but fell to 41.439
   TFLOPS. The RDNA 3.5 XML exposes only monolithic `S_BARRIER`, so the next
   credible path is to overlap the existing publication latency with an
   independent output tile or wave role while preserving two-workgroup LDS
   residency. Reject any design that increases scratch or loses full-output
   correctness.
3. **Separate package behavior from kernel behavior.** The 1,000-warm-up test
   slowed over sustained execution. Capture clocks and package power in a
   separate diagnostic pass, then keep the official timing pass free of
   counters and polling.
4. **Port the exact schedule into the PyTorch extension.** Preserve the pinned
   standalone harness as the baseline, then compare the integrated kernel with
   identical preallocated output and inputs before measuring API overhead.
5. **Keep promotion evidence reproducible.** The full reference must pass in
   every process and every process median must remain above 50.0 TFLOPS for
   the prepacked contract. Keep the ordinary-layout contract's separate gate.

## Low-priority paths for the square record shape

Do not repeat these unchanged without new evidence:

- The old `BLOCK_K=32` result no longer applies to the local interleaved K32
  experiment; `matmul_opt_k32` led that unmerged extension build. Do not repeat
  the old implementation unchanged or imply that this binding is in source
  base `8f7d926`.
- XOR swizzling was 15-20% slower.
- An LDS-staged vector C epilogue was 0.72x.
- The current ping-pong implementation was much slower and does not implement
  a useful producer/consumer specialization for GEMM.
- Split-K is useful for skinny or under-filled grids, but 4096 x 4096 with a
  128 x 64 tile already launches 2048 work-groups. Reduction overhead makes it
  a poor first choice for this record.
- A persistent work queue may help many small GEMMs, but launch overhead is not
  the main cost of one long 4096-cubed kernel.

FP16 accumulation may have a different speed/accuracy trade-off, but it is a
different benchmark class and must not replace an FP32-accumulation record.

## Latest frontier closure (2026-08-24)

The counter-guided follow-up kept the delta-2 hand-scheduled kernel as the
control: separate rocprofiler passes showed high active occupancy with waits
dominated by barriers, counters, and LDS retirement rather than DRAM transfer.
Removing the publish barrier produced a large numerical mismatch, while
streaming A fragments and changing WMMA issue order exceeded the register
budget (135 and 142 VGPR respectively). A persistent tile scheduler reached
35.821 TFLOPS despite exact output. Finally, moving the B refill before the A
refills preserved exactness and resources but measured 48.846/48.889/48.798/
48.690/48.784 TFLOPS across five fresh processes (48.801 average, 48.690
floor), so it is rejected. No unvalidated or isolated 50+ sample is promoted;
the delta-2 leader remains 49.035 TFLOPS average with a 48.980 floor.

The remaining compiled N-packed/N-major family was subsequently screened:
base 42.522, M-major 39.857, 2x4-warp 40.919, and K32 38.695 TFLOPS, all
exact but under the original-layout contract. A manual `s_setprio` cluster
hint preserved the delta-2 resource tuple but fell to 42.595 TFLOPS. An
`A0,B,A1` refill interleave reached 48.816 TFLOPS and was also rejected.

A critical-path LDS issue-order variant moved the first B fragment ahead of
the A fragments. It remained exact with the same 120-VGPR resource tuple, but
five paired candidate/control processes measured 49.076 versus 49.162 TFLOPS
on average (candidate floor 48.982 versus control floor 49.052). The initial
48.985-TFLOPS screen was package-state noise, so the original LDS order stays
the control.

Relaxing the hot-loop `lgkmcnt` waits by one produced a 49.117-TFLOPS timing
sample but failed correctness (`normalized_max_error=137.345592071`, cosine
`0.693088403`). The current wait thresholds are load-bearing and remain
unchanged.

### 2026-08-24 wave-range scheduling screen

The source block-prepacked kernel was rebuilt with tighter
`amdgpu_waves_per_eu` ranges to test whether occupancy metadata could improve
the producer/consumer schedule. All candidates passed the exactness tuple and
reported two active blocks/16 waves per CU. The 20-warmup/50-iteration medians
were 47.549 TFLOPS (1--2), 47.622 (2--2), and 47.608 (1--4), below the source
control and far below the hand-scheduled delta-2 image. Wave-range metadata is
therefore closed as an independent route to 50 TFLOPS.

### 2026-08-24 accumulator-bank permutation screen

The hand image was rebuilt with a cyclic permutation of the physical FP16
accumulator banks, leaving the v0-based epilogue bank fixed and rotating the
other three complete banks. Both non-identity rotations were rejected by the
gfx1151 loader (`hipOccupancyMaxActiveBlocksPerMultiprocessor` returned
`invalid device function`) before correctness or timing. A previously
validated pairwise bank swap still loads and runs, so this is specific to the
cyclic mapping rather than a general inability to change accumulator
placement. No TFLOPS result is recorded and the retained delta-2 image is
unchanged.

Removing only the pre-publish `lgkmcnt(0)` wait, while retaining the VMEM
wait and both barriers, remained exact but reached 48.904 TFLOPS. The wait is
not redundant on gfx1151 and remains part of the control sequence.

Moving the intermediate `lgkmcnt(2)` wait one WMMA later reached 49.073
TFLOPS but failed validation (`finite=no`, normalized maximum error
0.100156495, NaN RMS/cosine). The current wait placement is a real fragment
dependency boundary and remains unchanged.

Reversing each contiguous independent WMMA run preserved exactness and the
120-VGPR resource tuple but reached 48.809 TFLOPS. Matrix issue order is not
a free gain; the original order remains the control.

The remaining compiled B-phase/load backlog was screened: B-resident reached
44.600 TFLOPS exact, B-middle ordering 48.672 exact, and A-stripe ordering
failed correctness with normalized error 208.326. Two B-phase assembly objects
failed occupancy with invalid device functions before timing. None advances the
prepacked leader.

The prefetch-permutation backlog was screened as well: `bp-pf-*` forms ranged
from 47.308 to 47.649 TFLOPS, while the lower-occupancy `bp-prefetch-*` forms
reached 37.032–38.031 TFLOPS with three resident blocks. All valid outputs
were exact; none advances the delta-2 control.

VMEM clause/grouping variants were screened as well: `bp-clause2-break1`,
`bp-clause3`, `bp-no-clause`, `bp-no-explicit-vmwait`, and `bp-no-vmwait`
measured 47.723–47.445 TFLOPS, all exact at the same 120-VGPR occupancy.
Every form regressed from the delta-2 schedule.

Paired-fragment/read2 variants were screened too: B-pair forms reached
47.576–47.642 TFLOPS, p4 64-bit reads 44.103, and p8 read2 forms
43.753–43.809 TFLOPS. All valid output was exact, but none improved the
delta-2 control.

Indexed/static double-buffer forms reached 22.358–40.765 TFLOPS across their
padding variants, all exact. The extra LDS residency and index arithmetic lose
the leader's two-block occupancy, so this family is closed.

K32 ring proxies reached only 32.060–35.487 TFLOPS across minimal, reuse-B,
W8, and fenced forms, despite exact output. Their extra LDS traffic and
handoff work dominate, so extending the ring to K64 is not a promising direct
path.

Global-load cache-policy variants reached 46.362–47.351 TFLOPS for A/B
DLC/GLC/SLC combinations, with both-SLC at 46.560. All were exact, but cache
hints regress from the ordinary prepacked refill policy.

Direct-B/shared-A was also screened: it reached 32.803 TFLOPS with exact
output. Direct global B latency and duplicate wave traffic outweigh removing B
LDS reads, so the all-wave block-prepacked dataflow remains preferred.

Hoisting direct-B VMEM groups past four serialized waits improved that
architecture from 32.803 to 38.854 TFLOPS exactly (145 VGPR, 24 KiB LDS).
Removing the remaining `vmcnt(4)`/`vmcnt(2)` fences reached 39.039 TFLOPS but
failed correctness (`normalized_max_error=0.324106677`, cosine 0.992856290).
Direct-B remains latency-bound.

Pair-local A handoff variants reached 36.829–44.185 TFLOPS, all exact. Pair
signaling costs more than the retained workgroup barrier, so producer/consumer
handoff is closed for this square workload.

Wave-private/staging variants reached 35.850–37.492 TFLOPS; the 128×128 form
also failed the numerical threshold with normalized error 0.087310902. Lower
cooperative LDS sharing or occupancy is not competitive with delta-2.

A 1,000-warmup qualification of delta-2 measured 49.099 TFLOPS over five
timing blocks, with exact output. Extended warmup does not remove the
sustained gap to 50 TFLOPS.

The latest synchronization screen moved the final `lgkmcnt(0)` retirement wait
after the publish barrier. It retained 120 VGPR and 18 KiB LDS and reached
48.731 TFLOPS, but failed the full-output gate (`normalized_max_error=0.146452791`,
cosine `0.999337587`), confirming that wait-before-barrier is required for LDS
publication. Same-lock screens of the remaining prebuilt refill variants also
closed B-first (48.689), A0/B/A1 (48.662), scheduler priority (42.253), and
hybrid-A/B (43.768--44.479) forms. Every valid form was exact, but none
advanced delta-2.

A final register-pressure probe delayed the last B global load until after the
hot-loop WMMA group and reused lower B-fragment registers for the LDS store.
`bp-register-pack-b-after` faulted on its first launch with a GPU page-not-present
memory-access fault, before it could pass the differential gate. The same-lock
delta-2 control measured 48.718 TFLOPS and remained exact. The probe is closed:
the tail B fragment still requires the high VGPR range, so the transformation
did not lower the allocation class and introduced an unsafe live-range/issue
ordering.

The next dataflow screen moved the next B global load to the top of the K loop,
before the ten LDS fragment reads, and reordered VMEM retirement stores to match
the new completion order. This gave B the full WMMA window of latency hiding,
but an interleaved five-pair bracket measured 48.784 TFLOPS versus 48.823 for
the delta-2 control. A B-first late-load ordering was also exact but reached
48.525 TFLOPS. Both global-load orderings are closed as regressions.

The paired-K transfer screen measured the smallest supported FP16
block-prepacked geometry. A 128x128, four-wave tile with the native fragment
mapping compiled at 153 VGPR and 12 KiB LDS, passed the exact full-output tuple,
and reached 40.661 TFLOPS in a 100-warmup/100-iteration screen. A forced
eight-wave 128x128 mapping faulted before validation because it violates the
kernel's fragment/layout contract. Four-slot paired-K residency at 256x128
would consume about 48 KiB LDS per block and remove the two-block occupancy of
the delta-2 leader. The IU4 paired-K schedule is thus not a direct FP16 port;
the next candidate needs a different fragment partition.

A composition sweep shifted the B LDS base in the retained delta-2 assembly by
+4, +8, and +16 bytes. All candidates remained exact at 120 VGPR, but measured
38.059, 38.051, and 48.520 TFLOPS in one same-lock screen. The small shifts
expose a severe LDS bank-phase penalty; the 16-byte form still regresses the
control. The winning VGPR phase is not independently composable with a B-bank
shift.

The 256x256 supertile was compiled to test A reuse across a doubled N tile. It
used 119 VGPR, 22 SGPR, and 24 KiB LDS, but its 16-wave block admitted only one
resident block per CU. The candidate passed the complete exactness tuple and
reached 37.486 TFLOPS in a 100-warmup/100-iteration screen. Reusing A across N
does not repay the lost two-block residency; this supertile is closed.

The transposed 128x256 block was compiled as an eight-wave, two-block control.
It used 119 VGPR and 18 KiB LDS, passed the full exactness tuple, and measured
45.930 TFLOPS. A hand conversion of its refill loads to scalar-offset MUBUF
briefly timed above 50 TFLOPS, but failed the numerical gate with normalized
error 1.34 and cosine near zero. The transposed address/descriptor contract
must be regenerated from source; the apparent >50 result is rejected.

## Record protocol

Use this protocol before promoting a result in the README:

1. Record GPU model, firmware/driver, ROCm and PyTorch versions, compiler
   version, repository commit, power policy, clocks, and memory configuration.
2. Use the exact 4096 x 4096 x 4096 shape, FP16 inputs, FP32
   accumulation/output, fixed random seeds, and identical input tensors for all
   candidates.
3. Compare against `A.float() @ B.float()`. Report maximum absolute error,
   maximum error normalized by the reference maximum, and RMS error. Keep the
   historical normalized-maximum threshold below 1%, but also retain the raw
   metrics so the gate can be tightened later.
4. Decide whether the metric is kernel-only or end-to-end API time and state it.
   The current Python binding allocates a fresh FP32 output on every call. A
   kernel-only record should use a preallocated output or a direct launcher.
5. Run at least ten warm-ups and 100 timed iterations, repeated in five fresh
   processes. Report the median for each process plus the overall median and
   spread. Keep counter collection in separate runs because profiling changes
   execution timing.
6. Save machine-readable raw results and the generated ISA/resource metadata
   in the commit or release artifact.
7. Promote a new record only when the improvement is larger than normal
   run-to-run noise. The historical 21.6 TFLOPS record would require more than
   22.03 TFLOPS under a 2% noise margin. The ordinary-layout gate remains five
   fresh-process medians above 41.0 TFLOPS; the prepacked research gate is five
   fresh-process medians above 50.0 TFLOPS.

Current ROCm documentation lists gfx1151 as supported by ROCm Compute Profiler.
Counter availability still depends on the installed driver and profiler, so
query the target host and document unavailable counters instead of assuming
that all counters work or that none do.

### 2026-08-24 warp-tile-2 contract closure

The `warp_tile_m=2` fragment-partition screen was revisited at source level.
The first attempt had a launcher/header tile-size mismatch; after aligning both
to 128x128 and repairing the half-wave B-vector mapping, the kernel still
terminated with an unspecified launch failure before the exactness gate. It
has no valid timing result and remains closed until the fragment loader is
redesigned rather than patched in place.

### 2026-08-24 FP32-accumulator arithmetic screen

The same 256x128 block/K16-prepacked dataflow was rebuilt with the hardware
FP32-accumulator WMMA instruction and an explicit FP32-to-FP16 output
conversion. It retained two blocks/16 waves and passed the complete reference
tuple, but reached 47.922 TFLOPS in the source screen. The arithmetic change
therefore costs about 2.3% versus the 49.035-TFLOPS FP16-accumulator leader;
precision alone does not provide the missing headroom. The temporary source
override and binary were removed.

### 2026-08-24 A-major WMMA traversal screen

The prepacked 256x128 source loop was traversed A-major: each A fragment
visited all four B fragments before advancing to the next A fragment. This
changed accumulator dependency spacing while preserving the same LDS layout,
two-block/16-wave occupancy, and exactness tuple. The candidate reached 47.986
TFLOPS in the source screen, below the hand-scheduled 49.035-TFLOPS leader.
The alternate traversal is closed as a standalone gain.

### 2026-08-24 full-tile direct-store epilogue

Because the benchmark shape is exactly divisible by the 256x128 tile, a source
variant replaced the generic bounds-checked FP16 epilogue with unconditional
full-tile stores. It retained exact output and two-block/16-wave occupancy but
reached 47.850 TFLOPS, below the 49.035-TFLOPS leader. Edge-predicate removal
does not account for the remaining gap; the temporary variant was removed.

### 2026-08-24 source-level raw-buffer prefetch screen

The transposed refill path was regenerated from source with raw-buffer loads
and an explicit descriptor, rather than textual register substitution. It
preserved the full exactness tuple. On 128x256 it measured 45.897 TFLOPS versus
45.930 for the source control; on 256x128 it reached 47.307 versus 47.251.
The latter difference is within screening noise and neither geometry approaches
the 49.035-TFLOPS delta-2 leader. The temporary header and binaries were
removed after the run. Raw-buffer prefetch is closed as a standalone route.

### 2026-08-24 current-host requalification

The retained `bp-register-phase-d2` binary was requalified in five fresh
processes with full validation and 20 warmups/100 timed iterations. Medians
were 49.052, 48.828, 48.846, 48.724, and 48.680 TFLOPS (48.826 average).
Every process reproduced the exact reference tuple, but none reached 50. The
earlier 49.035-TFLOPS five-process qualification remains the stronger retained
record; this spread is the current control baseline for further assembly work.

### 2026-08-24 256x64 fragment geometry

An explicitly repaired 256x64 block was tested as a four-wave block with four
resident blocks per CU. The loader covered four A vectors per thread and one B
vector per thread, preserving the full exactness tuple. It reached 42.962
TFLOPS. The extra A traffic from splitting the N tile dominates the residency
benefit, so the half-width geometry is closed.

The complementary 256x192 geometry was rejected before benchmarking because
192 does not divide the fixed 4096-wide problem. Its edge workgroup reached
outside the packed/output bounds and faulted; this is a launch-contract failure,
not a throughput measurement.

### 2026-08-24 SGPR phase screen

The delta-2 assembly was rebuilt with scalar registers at two aligned physical
phases while staying inside the same 32-SGPR allocation class. The +4 phase
used 26 SGPR and remained exact, but five medians averaged approximately 48.73
TFLOPS. The +8 phase used 30 SGPR and also remained exact; medians were
49.099, 49.172, 49.083, 49.030, and 48.880 TFLOPS (49.053 average, 48.880
floor). These results track package variation rather than a separated gain,
so SGPR placement is closed as a standalone optimization.

### 2026-08-24 A-side LDS phase screen

The hand assembly’s A LDS base was shifted consistently in both initial and
refill stores/loads. The +8-byte phase remained exact but collapsed to 35.843
TFLOPS; the +16-byte phase remained exact with medians 48.958, 49.145, 48.962,
48.928, and 48.989 TFLOPS (about 48.997 average). A-bank phase is load-bearing
but offers no sustained gain; the B/A phase-composition branch is closed.

### 2026-08-24 combined A/B LDS phase

Shifting both A and B LDS accesses by +16 bytes preserved the exact output and
the 120-VGPR/two-block resource tuple. Its five fresh-process medians were
48.649, 48.686, 48.775, 48.822, and 48.698 TFLOPS (about 48.726 average),
below the control band. The operands’ bank phases do not combine into a gain;
the phase-composition branch is closed.

### 2026-08-24 paired A-LDS issue order

The A LDS transactions within each fragment group were issued in ascending
bank-offset order while preserving destination registers and all waits. The
variant stayed exact at the same 120-VGPR/two-block resource tuple, but fresh
process medians averaged about 48.629 TFLOPS. Request ordering is therefore not
an independent gain; the original hand schedule remains the control.

### 2026-08-24 A-fragment register permutation

The A0/A1 fragment register groups were swapped consistently in loads, refill
stores, and WMMA operands, changing physical operand-bank placement without
changing dataflow or resources. The corrected candidate passed exactness at
120 VGPR/two blocks, but five medians averaged about 48.855 TFLOPS. A-fragment
register placement alone is closed.

The complementary A2/A3 register-group permutation was also exact at unchanged
resources, but five medians averaged about 48.752 TFLOPS. Together with the
A0/A1 result, this closes A-fragment register permutation as a standalone path.

The final K-tile B0/B1 register groups were then swapped consistently through
their LDS loads and WMMA operands. The exact candidate retained 120 VGPR and
two-block occupancy, but five medians averaged about 48.711 TFLOPS. The
49.585-TFLOPS short sample was package noise; final-tile B placement alone is
closed.

### 2026-08-24 block traversal screen

The prepacked kernel's mapping layer exposes Morton and CU-oriented traversal
modes in addition to the default XOR-snake order. Source controls at modes
1, 2, 5, and 6 passed the complete exactness tuple, but their
20-warmup/50-iteration medians were 46.438, 46.463, 46.136, and 45.853 TFLOPS.
Alternate traversal therefore did not recover the hand-scheduled leader's
gap.

A separate fixed-shape experiment replaced the generic reciprocal/XOR mapper
in the hand assembly with a 4096-square row-major shift and mask. The code
object was rejected by the gfx1151 loader before the occupancy query could
run; it produced no timing or correctness result. This is a launch-contract
failure, not evidence for or against row-major locality. Temporary assembly
and binaries were deleted, and the existing mapping and delta-2 schedule are
unchanged.

### 2026-08-24 composed accumulator-bank permutation

The earlier cyclic rewrite was repeated using the repository's register-aware
pair-swap transform twice, producing a valid three-bank cycle while preserving
the v0-tied epilogue bank. It passed the complete exactness tuple and loaded at
the same 120-VGPR/two-block occupancy. A five-process, 20-warmup/100-iteration
qualification measured 48.839, 48.698, 48.749, 48.649, and 48.712 TFLOPS
(48.729 average, 48.649 floor). The first 49.059 short screen was therefore
package noise; the composed permutation is not promoted over delta-2.

The opposite three-bank cycle was also constructed by composing the pair-swap
transform (17<->49 followed by 17<->33). It was loader-valid and exact, but
its five-process qualification measured 48.782, 48.721, 48.642, 48.722, and
48.733 TFLOPS (48.720 average, 48.642 floor). Both cycle directions are
therefore closed; accumulator placement has no measured path to 50 TFLOPS.

### 2026-08-24 SALU-in-WMMA-window screen

The independent loop-counter decrement was moved from immediately before the
first WMMA issue into the first WMMA group, leaving all LDS waits, barriers,
fragments, and branch semantics unchanged. The image stayed exact at
120 VGPR/two-block occupancy. Five fresh processes measured 48.777, 48.740,
48.669, 48.624, and 48.788 TFLOPS (48.720 average, 48.624 floor), so the
49.094 short screen was package variation and the SALU placement is closed.

### 2026-08-24 K32 stage/ring screen

The source K2 specialization (`WMMA_BP_K_SLICES=2`) was first run with a
host-packer mismatch: the kernel used K32 while `RECORD_K_SLICES` remained 1,
so its 44.350-TFLOPS result failed exactness and was invalid. Rebuilding with
both kernel and host packing set to K32 produced an exact 44.289 TFLOPS image
at 120 VGPR/two-block occupancy. Rebuilding the dedicated K2 ring with the same
correct host packing instead failed exactness (normalized error 1.365675535,
cosine -0.000019926) behind a 42.198-TFLOPS timing. Its earlier 41.345-TFLOPS
“exact” result used the mismatched K16 host layout and is invalid. Correct K32
packing repairs only the generic path, which remains below delta-2.

The corrected K2 path with `WMMA_BP_NO_EXPLICIT_VMWAIT=1` stayed exact but
reached 44.321 TFLOPS, statistically identical to the 44.289 control. Removing
the explicit VMEM wait does not recover the K32 handoff cost.

An alternate K32 slice-major host packing (`RECORD_BLOCK_SLICE_MAJOR=1`) was
also tested with the corrected K2 kernel. It reached 44.424 TFLOPS but failed
the full reference (normalized error 1.356285863, cosine -0.000246312), so the
kernel’s address contract is specifically block/K-major. Slice-major packing
is rejected.

The retained delta-2 control was rechecked after the corrected K32 screens on
the current host: 48.911 TFLOPS in a 20-warmup/100-iteration five-block run,
exact output. This is consistent with the established 48.980--49.035
qualification band and does not change the promotion gate.

### 2026-08-24 MAC-priority window screen

The next producer/consumer hypothesis was a scheduler-policy change rather
than another LDS layout: emit `s_setprio 1` around the hand-scheduled WMMA
cluster and restore `s_setprio 0` immediately before each LDS handoff. The
transformation is reproducible with `tools/patch_setprio_asm.py` and leaves
the delta-2 resource tuple unchanged (120 VGPR, 22 SGPR, 18 KiB LDS).

The gfx1151 runtime rejected the resulting code object before validation:
`hipOccupancyMaxActiveBlocksPerMultiprocessor` returned `invalid device
function` in all five isolated launches. A source-level priority build (not
the hand-scheduled leader) was loadable and exact but only reached
41.84--42.26 TFLOPS, so it is not a comparable promotion. The hand-scheduled
priority window is closed as an unsupported ISA/runtime path; no invalid
timing is retained.

### 2026-08-24 late-refill double-buffer screen

The previously unqualified `WMMA_BP_DOUBLE_BUFFER_LATE=1` branch was built on
the same 256x128 block/K16-prepacked input contract. It keeps complete A and B
tiles in two LDS buffers and refills the inactive buffer later in the WMMA
cluster. The image stayed exact in all five isolated launches, with the same
normalized maximum error `0.018779343`, RMS `0.035428338`, and cosine
`0.999977929` as the control.

The extra LDS allocation reduced occupancy to one block/eight waves per CU.
The five medians were **38.987, 39.192, 38.978, 39.197, and 38.856 TFLOPS**
(39.042 average, 38.856 floor), so late refill cannot trade synchronization
for enough residency and is closed for the square record shape.

### 2026-08-24 split-barrier and wait-order screen

The proposed split producer/consumer barrier (`s_barrier_signal` /
`s_barrier_wait`) was attempted on the packed path, but ROCm 7.14's gfx1151
assembler rejected both instructions as unsupported. No image was produced.
The related loadable schedule moved the VMEM wait after the workgroup barrier
(`WMMA_BP_WAIT_AFTER_BARRIER=1`). It passed the exact tuple in five launches,
but measured **47.746, 47.619, 47.633, 47.385, and 47.335 TFLOPS** (47.543
average, 47.335 floor). The existing wait-before-barrier ordering remains
strictly better; split signaling is an ISA dead end and wait-after-barrier is
closed as a regression.

### 2026-08-24 hybrid-A grouped producer screen

The hybrid A ping-pong branch was rebuilt with grouped A loads and a late A
commit (`WMMA_BP_HYBRID_A_PINGPONG=1`, `WMMA_BP_HYBRID_A_GROUP_LOADS=1`,
`WMMA_BP_HYBRID_A_LATE_COMMIT=1`). This keeps B single-buffered while A is
written to an inactive LDS buffer, providing a distinct producer/consumer
schedule without the full two-operand double-buffer footprint. All five
launches passed the exact tuple, but medians were **44.551, 44.539, 44.333,
44.391, and 44.246 TFLOPS** (44.412 average, 44.246 floor). The extra A
handoff work outweighs the overlap; this hybrid schedule is closed for the
square record shape.

### 2026-08-24 late-B1 128x128 ownership screen

The late-B1 prefetch specialization was built with its required 128x128
single-buffer tile (`WMMA_BP_LATE_B1_PREFETCH=1`, two M-waves and two N-waves).
All five launches passed the full exactness tuple and reported four active
blocks/16 waves per CU. Medians were **41.166, 40.840, 40.764, 40.721, and
40.902 TFLOPS** (40.879 average, 40.721 floor). The smaller tile's additional
block traversal and refill work outweigh the late B overlap; this ownership
specialization is closed well below the 256x128 leader.

### 2026-08-24 fixed-tile epilogue probe

The hand-scheduled delta-2 tail was specialized for the exact 4096x4096
benchmark by removing the late N-fragment edge compares while retaining the
existing stores. The candidate briefly timed at **49.117 TFLOPS**, but failed
the full-output gate (`normalized_max_error=0.848721961`, RMS `2.167147235`,
cosine `0.913690612`). A second form removed the corresponding exec masks and
branches entirely; the gfx1151 runtime rejected that image as `invalid device
function` before validation. The apparent gain is rejected: these predicates
are entangled with the generated exec state, and fixed-shape specialization is
not a safe route without redesigning the epilogue from source.

### 2026-08-24 source full-tile store specialization

The epilogue was then redesigned in source behind
`WMMA_BP_FULL_TILE_STORE=1`. The opt-in helper writes the N-packed fragments
without per-element bounds checks while retaining the normal kernel exec state
and fragment mapping; it is statically restricted to the 256x128 record
geometry. All five launches passed the exact tuple and retained two
blocks/16 waves, but medians were **47.749, 47.805, 47.555, 47.730, and
47.558 TFLOPS** (47.679 average, 47.555 floor). Source-level full-tile stores
are therefore correct but slower than the hand-scheduled delta-2 leader and
are closed as an independent path.

### 2026-08-24 fresh delta-2 counter pass

A separate ROCm Compute Profiler pass on the retained `bp-register-phase-d2`
image collected barrier/LDS counters without using serialized counter timing
as a throughput denominator. `SQ_BUSY_CYCLES_avr / GRBM_GUI_ACTIVE` was
0.9904, confirming that the kernel is shader-active rather than DRAM-latency
bound. Normalized to `SQ_WAVE_CYCLES_sum`, the counters reported approximately
**17.2% `SQ_WAIT_BARRIER`, 5.7% `SQ_WAIT_INST_LDS`, and 8.6%
`SQ_WAIT_CNT_ANY`**. The next architecture target is therefore to hide or
remove the two workgroup handoffs without adding LDS capacity or register
pressure. Raw profiler output remains on the host at
`/root/wmma-results/profile-delta2-new/`.

### 2026-08-24 two-producer ring screen

The first two-producer launch deadlocked because the harness still launched
five waves: its default `RECORD_PRODUCER_WAVES=5` mismatched the kernel's two
producer plus four consumer waves. After rebuilding with six launched waves,
the ring was exact and reported three active blocks/18 waves per CU. Three
fresh processes measured **13.070, 12.746, and 12.800 TFLOPS** (12.872 average),
far below the packed leader. The architecture is now correctly closed as a
throughput regression; the initial deadlock was a harness error, not a kernel
result.

The complementary wide ring (`WMMA_PC_WIDE=1`, one producer plus eight
consumers, nine launched waves) was also exact but reached only **6.517 TFLOPS**
in a 5-warmup/5-iteration screen at three blocks/27 waves per CU. The extra
consumer polling and narrower per-wave N tile make this architecture clearly
noncompetitive; no longer promotion run is warranted.

### 2026-08-24 K-loop SCC shortcut

The hand assembly's K-loop decrement was tested as the source of the back-edge
condition, removing the explicit `s_cmp_eq_u32 s6, 0` before
`s_cbranch_scc0`. The assembler accepted the image, but the gfx1151 runtime
rejected it as `invalid device function` during the occupancy query. No
correctness or timing result exists; the explicit compare remains required by
the loader/code-object contract.

### 2026-08-24 epilogue address recurrence

The first eight scalar C stores were rewritten to carry a 64-bit output
pointer in two otherwise-unused VGPRs and advance it by the constant `4*N`
byte stride. Two VOP3 carry-in operand orderings were assembled, but both
images were rejected by the gfx1151 runtime as `invalid device function` at
the occupancy query. No timing or correctness result exists; the existing
per-store address reconstruction remains the supported hand-assembly form.

### 2026-08-24 pre-barrier VMEM wait screen

The first refill `s_waitcnt vmcnt(2)` was moved ahead of the publish barrier,
with the post-barrier copy of that wait removed. The assembled image was
rejected by gfx1151 as `invalid device function` during the occupancy query;
there is no correctness or timing result. The existing barrier-then-VMEM-wait
ordering is therefore a code-object boundary as well as the measured control.

### 2026-08-24 hybrid-B ping-pong screen

The complementary one-sided producer was built with B ping-pong and A left in
the active single buffer (`WMMA_BP_HYBRID_B_PINGPONG=1`) on the 256x128 packed
geometry. It remained exact in all five launches at two blocks/16 waves per
CU. Medians were **45.090, 45.107, 44.805, 44.976, and 44.742 TFLOPS**
(44.944 average, 44.742 floor). B-side overlap is no better than the grouped
A-side schedule and is closed as a standalone route to 50.

The companion `WMMA_BP_HALF_SWIZZLE=1` layout was also screened on the same
256x128 packed shape. It passed the complete exactness tuple at unchanged
two-block occupancy, but reached only 38.402 TFLOPS. Half-word LDS swizzling is
therefore closed as a standalone data-movement optimization.

### 2026-08-24 early-B prefetch probe

The next-B global vector was moved ahead of the second WMMA group and placed
in spare high VGPRs to lengthen its latency-hiding window. The first image
declared only 120 VGPR despite using v120:v123; its 51.372-TFLOPS timing was
invalid and failed exactness (normalized error 1.002724390). Raising the image
metadata to 128 VGPR and correcting VMEM completion order still failed with
the same error (49.575 TFLOPS in the copy-back form). The fast timing is not a
result; this register/live-range producer rewrite is rejected.

Adding an explicit `vmcnt(0)` before the high-register copy did not repair the
same mismatch (49.274 TFLOPS). The failure is therefore not merely a copy
ordering hazard; the high-register early-B image is closed.

The early-B artifact was regenerated cleanly after finding that an intermediate
rewrite had accidentally removed the relocated load. The corrected image did
contain one `buffer_load_b128 v[120:123]`, declared 128 VGPR, and still failed
the exactness gate at 49.886 TFLOPS. This confirms the producer is not rescued
by the earlier script correction; no timing from this family is promotable.

The complementary early-A staging image moved A vectors into v120:v127 and
declared 128 VGPR. It hung before returning occupancy or validation and was
terminated under the exclusive-run watchdog; no timing or correctness result
is recorded. This is a synchronization/launch failure, not evidence of an
A-prefetch benefit.

### 2026-08-24 mixed global-load clause screen

The hot-loop `s_clause` was extended from the two A loads to the A/A/B global
load trio, with no address, register, or wait changes. The candidate stayed
exact at 120 VGPR/two-block occupancy. Five fresh processes measured 48.961,
48.858, 48.840, 48.854, and 48.946 TFLOPS (48.892 average, 48.840 floor),
below delta-2; the 49.103 short screen was package variation. Clause grouping
is closed as an independent gain.

### 2026-08-24 A-load width probe

The two adjacent A `buffer_load_b128` instructions were replaced with a single
`buffer_load_b256` feeding the same eight VGPRs. The gfx1151 assembler rejected
the vector form (`invalid instruction`, suggesting only scalar
`s_buffer_load_b256`), so no device image, correctness result, or timing exists.
The instruction-width fusion is an ISA-level dead end.

### 2026-08-24 warp-tile ownership screen

The source packed kernel was rebuilt with `warp_tile_m=2` to reduce per-wave
M-fragment residency. With the existing four M-waves the derived geometry
became 128x128 and produced an exactness failure at 23.282 TFLOPS. An
eight-M-wave rebuild restored the 256x128 geometry but hit an unspecified
launch failure before validation. The alternative tile-ownership architecture
is closed without a performance result.

The opposite ownership geometry (`warp_tile_m=8`, two M-waves) was then
tested. With two N-waves it produced a non-finite result despite an invalid
86.892-TFLOPS timing. Splitting N into four waves (`warp_tile_n=2`) repaired
exactness at 47.590 TFLOPS and two-block/16-wave occupancy, but remains below
delta-2. Larger M stripes therefore do not provide a usable 50-TFLOPS path.

The source arrays were then resized (`c_n` and every packed-path `a_frag`)
to match `warp_tile_m=8`, eliminating the known four-entry out-of-bounds bug.
The repaired image still returned non-finite output behind an invalid 87.047
TFLOPS timing at three-block/12-wave occupancy. The failure is deeper than
array sizing (fragment/epilogue assumptions), so large-M ownership is closed.

The source paired-B path (`WMMA_BP_B_PAIR=1`) was screened on the retained
256x128 packed geometry. It stayed exact at 120 VGPR/two-block occupancy but
reached only 47.698 TFLOPS in the 20-warmup/50-iteration screen. Pairing B
loads/consumers is below delta-2 and is closed for this shape.

### 2026-08-24 native LDS load-width screen

Forcing `WMMA_NATIVE_LOAD_BITS=64` changed each native half fragment load from
128-bit to two 64-bit transactions. The packed 256x128 image stayed exact at
120 VGPR/two-block occupancy but reached only 41.618 TFLOPS. The wider native
load is not the missing bottleneck; 64-bit LDS loads are closed.

### 2026-08-24 streamed-B barrier screen

Enabling `WMMA_BP_STREAM_B_BARRIER=1` inserted scheduler barriers between B
fragment groups on the packed 256x128 source path. The output stayed exact, but
the image reached only 44.802 TFLOPS despite reporting three active blocks/24
waves per CU. The extra scheduling fences dominate; streamed-B barriers are
closed.

### 2026-08-24 asymmetric LDS padding screen

Complementary A/B stride pairs were tested on the packed 256x128 source
kernel: (4,12), (12,4), (2,14), (14,2), (6,10), and (10,6). Every candidate
passed the exactness tuple and retained two-block occupancy, but the
20-warmup/20-iteration screens reached only 25.755--27.389 TFLOPS. The
operand stride phases are coupled by the WMMA/LDS access pattern; asymmetric
padding is closed as a route to the leader.

### 2026-08-24 LDS wait-threshold screen

The counter pass suggested that the first WMMA pair might not need to wait for
all ten outstanding LDS returns. Hand-patched copies of the delta-2 image
lowered one dependency threshold at a time, with the same registers, addresses,
barriers, and input contract. Lowering the first `lgkmcnt(6)` to 5 remained
exact, as did lowering the second 4 to 3 or the third 2 to 1. Short 10/10
screens reached 49.930, 49.807, and 49.858 TFLOPS respectively, but all
five-process/longer interleaved tests stayed below the promotion gate. The
best single-step candidate (third threshold 2 to 1) averaged 48.807 TFLOPS
versus 48.565 for its interleaved controls; the first-threshold candidate
averaged 48.952 versus 48.893 for controls. The apparent gains are therefore
within package noise, while the combined 4/3/1 ladder fell to 49.597 in the
short screen. Retain the original 6/4/2 ladder; wait-threshold relaxation is
closed as a standalone route to 50 TFLOPS.

### 2026-08-24 direct packed-store probe

The fixed-shape epilogue was probed for a no-LDS vectorization: two adjacent
FP16 accumulator registers were packed with `v_pack_b32_f16` and emitted with
one `global_store_b32`. The image loaded at unchanged two-block/16-wave
occupancy, but the full output gate failed (normalized maximum error
0.767344810, RMS 0.668158748, cosine 0.992118715). The apparent register
adjacency is not the output lane adjacency; direct store coalescing is rejected
without a complete fragment-lane remap.

### 2026-08-24 normal 128x256 ownership screen

The complementary N-wide ownership geometry was rebuilt from the packed source
with a 128x256 block (2 M waves × 4 N waves), rather than the retained 256x128
tile. It used two resident blocks/16 waves and passed the full exactness tuple,
but reached **46.809 TFLOPS** in a 10-warmup/10-iteration screen. The smaller
M stripe loses reuse and cannot repay its additional N-fragment issue work;
normal 128x256 ownership is closed below delta-2.

### 2026-08-24 hand-scheduled mapping-swizzle screen

The delta-2 register-phase schedule was regenerated with the source workgroup
traversal period changed from 16 to 8 and 32, leaving the tile shape, register phase,
waits, barriers, and input contract unchanged. Both images passed the complete
exactness tuple at two blocks/16 waves. Short screens reached 49.410 and
49.439 TFLOPS respectively, but three fresh interleaved pairs for swizzle 32
averaged 48.771 TFLOPS versus 48.994 for delta-2 controls. The alternate bank
traversals regress under sustained timing; period 16 remains required.
The zero-padding source layout was also rebuilt as a control for the hand
schedule. The register-phase patch could not be applied because its descriptor
pattern is specific to the p8 address schedule; the loadable source image
passed exactness at two blocks/16 waves but reached only **45.272 TFLOPS** in
a 10/10 screen. Zero padding is not a route to 50 and is not mixed with the
hand-scheduled p8 record.

An A8/B0 mixed-padding source control was also attempted. The hand patcher
rejected it because the descriptor rewrite is tied to the p8 address pattern;
the unpatched source image remained exact at two blocks/16 waves but reached
44.993 TFLOPS in a 10/10 screen. Removing B padding without a new hand
descriptor is therefore closed.

The corrected DPP vector epilogue was then combined with the delta-2 hand
schedule. The source epilogue allocates a different 117-VGPR register map, so
the existing progressive-refill patch has no matching handoff. A direct
two-register phase shift was attempted with the adjusted allocation, but the
gfx1151 assembler rejected dual-VALU operands for VGPR bank conflicts. No
combined image was loadable; the standalone hand schedule and source DPP path
remain separate.

As a follow-up, a delta-4 phase shift (the smallest shift preserving the DPP
dual-VALU bank pairing) assembled across safe boundaries 33--81. The images
were exact but moved to three blocks/24 waves and measured 41.106--41.383
TFLOPS in 5/5 screens. DPP register shifts therefore do not approach the hand
leader; the unshifted source DPP image remains the only useful reference.

The 4096-square record was also given a fixed-grid mapper that replaces the
general swizzle/division prologue with direct `blockIdx.x` row/column shifts.
The row-major image stayed exact at 120 VGPR/two-block occupancy and briefly
reached 49.596 TFLOPS, but fresh interleaved timing averaged 48.536 TFLOPS
versus 48.773 for controls. A hand reconstruction of the default XOR-snake
map with scalar bit arithmetic produced incorrect output (normalized error
1.000000), so fixed traversal is closed without a verified equivalent cache
mapping.

The repository's known `cu_5x8_record_mapping` (mapping mode 5) was also
screened. Its source image was exact at 46.933 TFLOPS, but splicing only its
prologue into delta-2 was invalid because the mapping changes the pointer/SGPR
contract after the mapper. A source-only delta-2 register shift remained exact
but raised allocation to 121 VGPR/three blocks and reached 38.857 TFLOPS. The
5x8 traversal is not a drop-in hand-schedule optimization.

### 2026-08-24 fixed column-major traversal and paired-K IU4 qualification

The last low-cost FP16 mapper alternative reversed the exact-grid traversal to
column-major order while keeping the delta-2 hand schedule, pointer contract,
and two-block occupancy unchanged. It passed the full 4096-cubed reference
tuple, but measured **44.403 TFLOPS** (5 warmups/5 iterations). The cache order
is therefore material to the hand schedule; this variant is closed rather than
promoted.

The separate `v_wmma_i32_16x16x16_iu4` architecture was then requalified with
the paired-K prepacked W4A4 GEMM. Five fresh candidate/control pairs used 10
warmups and 10 timing iterations per process. Every candidate and control had
zero mismatches over all 16,777,216 INT32 outputs. Candidate medians were
92.298, 90.195, 91.040, 90.105, and 91.085 INT4 TOPS (average **90.945**);
the one-slice controls were 83.861, 84.621, 84.484, 84.867, and 84.969 (average
84.560), a 7.6% paired-K gain. This is a real exact integer-WMMA result, but it
is not an FP16 TFLOPS result and cannot be substituted for the 49.035 FP16
leader without changing the activation/weight and scale contract.

An eight-slice ring (`-DIU4_PAIR_K=4`, 16 rotating LDS slots, 32 KiB LDS per
block) remained exact at 4096 cubed but reached only **92.056 INT4 TOPS** in a
10-warmup/10-timing screen. The extra resident state does not repay its LDS
traffic and occupancy pressure; four slices remain the selected IU4 depth.

### 2026-08-24 later-LGKM wait removal screen

Each of the four later `s_waitcnt lgkmcnt(0)` points in the delta-2 hand loop
was removed independently, leaving all addresses, WMMAs, barriers, and other
waits unchanged. Every assembled image was rejected by gfx1151 as
`invalid device function` during the occupancy query, before correctness or
timing. These waits are therefore code-object/synchronization boundaries, not
safe overlap opportunities; the delta-2 wait schedule is retained.

The paired-K ring was then generalized from two to four resident K16 slices
(`-DIU4_PAIR_K=2`), using eight rotating LDS slots and one publication per
four slices. A second fresh five-process bracket (10 warmups/10 timings per
process) stayed exact over all 16,777,216 INT32 outputs. Candidate medians were
94.141, 94.004, 93.445, 94.700, and 93.931 INT4 TOPS (average **94.044**,
floor 93.445); the two-slice controls were 91.640, 91.208, 91.514, 91.397,
and 90.460 (average 91.244). The earlier bracket contained one 84.765 TOPS
candidate outlier and is not used for promotion. Four-slice residency is now
the best qualified IU4 GEMM, but remains separate from the 49.035 FP16 goal.

### 2026-08-24 instruction-fetch phase and WGP placement screens

The retained delta-2 K loop starts at device offset `0x5b0`, with its first
WMMA at `0x608`. A bounded assembly sweep moved the loop header to the next
64- and 128-byte boundaries and sampled the other 64-byte instruction-fetch
phases with one-time `s_nop` padding before the back-edge target. No padding
was inserted in the repeated K-loop body. Every image retained 120 VGPR,
22 SGPR, 18 KiB LDS, two blocks/16 waves, and reproduced the exact reference
tuple. The phase candidates reached 49.355--49.677 TFLOPS between 49.444 and
49.694-TFLOPS controls; the spread followed run order rather than fetch phase,
and no candidate crossed 50 TFLOPS. Fetch placement is closed as a standalone
gain. `build-hot-loop-align.sh`, `tools/align_hot_loop_asm.py`, and
`tools/pad_hot_loop_asm.py` reproduce the screen.

The same instruction image was then changed from CU to WGP placement only in
its code-object descriptor. WGP mode remained exact and exposed three active
blocks/24 waves per CU according to the occupancy query, but reached only
**47.406 TFLOPS** between exact CU controls at 49.442 and 49.526 TFLOPS.
Distributing an LDS-sharing eight-wave workgroup over all four SIMD32s adds
more synchronization/data-sharing cost than its extra residency repays. The
49.035-TFLOPS qualified leader therefore remains in CU mode.
`build-wgp-mode.sh` and `tools/set_workgroup_mode_asm.py` preserve this
negative result as a reproducible placement experiment.

### 2026-08-24 LDS-only publication wait qualification

The repeated delta-2 handoff already retires all three refill VMEM operations
at `vmcnt(2)`, `vmcnt(1)`, and `vmcnt(0)` before the final B LDS store. Its
following combined `s_waitcnt vmcnt(0) lgkmcnt(0)` therefore re-tests a VMEM
counter known to be empty. `tools/patch_publish_wait_asm.py` narrows only this
hot-loop instruction to `s_waitcnt lgkmcnt(0)`; the prologue, addresses,
WMMAs, stores, barriers, register allocation, and metadata remain unchanged.

Six 20-warmup/100-iteration pairs alternated candidate/control order. The
candidate medians were **49.376, 49.307, 49.149, 49.029, 49.032, and 48.964
TFLOPS** (49.143 average, 48.964 floor), while paired delta-2 controls were
49.307, 49.065, 49.010, 48.833, 48.918, and 48.950 TFLOPS (49.014 average).
The candidate won all six pairs by 0.129 TFLOPS on average (+0.26%), retained
120 VGPR/22 SGPR/18 KiB LDS and two blocks/16 waves, and reproduced the exact
reference tuple in every process. It supersedes the 49.035-TFLOPS qualified
average as the block/K16-prepacked FP16 research leader, but no process crossed
the 50-TFLOPS promotion threshold. Raw process output is retained on the GPU
host at `/root/wmma-results/early-barrier-lgkm-20260824.txt`.
Its SHA-256 is
`fb90cec6d232d17fcc0fe933160f4684bf81e0617a3bc433683c5069353b4cfa`.

The counter boundary is strict. Allowing one or two LDS operations to remain
outstanding produced normalized maximum errors 0.069077660 and 0.073682838,
respectively, above the 0.03 gate; those timings are invalid. A separate
CU-local architecture moved the overwrite barrier before the final three
register-resident WMMAs and issued the refill stores before or between them.
All loadable forms were exact, but a six-pair order-balanced isolation measured
48.866 TFLOPS for the best early-barrier form versus 48.890 for the equivalent
LDS-only-wait image. Moving the WMMAs contributes no independent gain and is
retained only as negative architecture evidence. `build-publish-wait.sh` and
`build-early-read-barrier.sh` reproduce both families.

### 2026-08-24 loop control, ISA XML, and compact-tile architecture

The fixed K=4096 loop was peeled once and unrolled by two with
`tools/unroll_kloop_asm.py`, removing one compare and back-edge branch per two
K16 slices without changing any load, wait, WMMA, store, or barrier. The image
kept the leader's 120-VGPR/22-SGPR/18-KiB resource tuple and exact full-output
error tuple. Six alternating-order 20-warmup/100-iteration pairs averaged
48.798997 TFLOPS for the unroll and 48.803105 for the LDS-only-wait control;
each won three pairs. Scalar loop control is therefore neutral, not the
remaining bottleneck. Raw output is
`/root/wmma-results/kloop-unroll2-20260824.txt` (SHA-256
`43113cbe964704191cb4957ef64000a6915d061286fd1150dd092f95af446faf`).

ISA decisions in this pass were checked against AMD's machine-readable
RDNA 3.5 XML, downloaded from
[`gpuopen.com/download/machine-readable-isa/latest/`](https://gpuopen.com/download/machine-readable-isa/latest/).
On 2026-08-24 that endpoint resolved to
`AMD_GPU_MR_ISA_XML_2026_08_06.zip`; the relevant member is
`amdgpu_isa_rdna3_5.xml`. It exposes only the monolithic `S_BARRIER` on this
generation, not split signal/wait barriers, and the widest LDS vector
operations are `DS_LOAD_B128` and `DS_STORE_B128`. This independently closes
split-barrier and 256-bit LDS-transfer proposals, matching the ROCm 7.14
assembler behavior. Future instruction claims should use this XML as the
primary ISA inventory rather than borrowing gfx12 behavior.

The code-object scheduler declarations were also isolated with identical
device instruction streams. Setting `FWD_PROGRESS=0` (oldest-first),
`MEM_ORDERED=0`, or both produced exact short medians of 49.454, 49.590, and
49.486 TFLOPS, respectively, bracketed by 49.586 and 49.776 controls. The
oldest-first policy regressed and unordered memory reporting was neutral; no
long qualification was justified. Raw output is
`/root/wmma-results/descriptor-modes-20260824.txt` (SHA-256
`1448fe1d98b97aaf4aa215058a2f023a3114e2dfbddffa74ae22dfc8a3a40a80`).

Finally, the dormant 128x128 `warp_tile_m=2` path was audited as a genuinely
different ownership architecture. Its initial load covered only half of B,
its generic 128-row fallthrough issued out-of-bounds `2*tid` vectors, and its
steady-state prefetch repeated the mismatched mapping. The opt-in
`WMMA_BP_WARP_TILE2_REPAIR` gives each of the 256 threads one 16-byte A vector
and one 16-byte B vector in the initial load, LDS publication, and every K16
refill. The repaired kernel is exact over all 16,777,216 outputs and compiles
to 90 VGPR, 22 SGPR, 12 KiB LDS, no spills, and four blocks/32 waves per CU.

That extra residency did not win. The first repaired screen reached 44.568
TFLOPS. Moving the two refill issue points across the four N steps kept every
candidate exact and produced 44.376 (`a0b0`), 44.441 (`a0b1`), 44.358
(`a0b2`), 44.661 (`a0b3`), and 44.796 TFLOPS (`a1b0`). Halving useful WMMA
work per wave doubles block-level publication boundaries for the same matrix;
32 resident waves do not repay that synchronization rate. The compact path
is retained as a correct architectural control, not promoted or long-qualified.
Raw outputs are `/root/wmma-results/warp-tile2-repair-20260824.txt` (SHA-256
`8fee79bacb254f8ec3692661a5e6a79aef3192f70220e390eba81ef19fe73f17`) and
`/root/wmma-results/warp-tile2-prefetch-20260824.txt` (SHA-256
`c8fb1b74fdae35eb8085e4f297e7789da9dc814c1f8389784e086d5e41e9147a`).

The direct follow-up amortized two K16 slices in one compact K32 publication.
The guarded path kept the repaired 128x128 ownership, staged two contiguous
vectors per thread for both operands, and performed both WMMA slices before
each overwrite handoff. It was full-output exact and admitted three blocks/24
waves per CU, but the K32 state raised resources to 122 VGPR and 20 KiB LDS.
Its 20-warmup/5x10 screen reached only 43.902 TFLOPS. Halving the barrier count
does not repay the wider LDS stride and extra staging/fragment state, so compact
K32 is also closed without qualification. Raw output is
`/root/wmma-results/warp-tile2-k32-20260824.txt` (SHA-256
`c2331328ae3ef595c4cbe5607871d665032134928b57e6f630644519147b9150`).

The next hand schedule used the otherwise-free allocation tail `v120:v127` to
double-buffer B fragments. It issued B1 with the ten initial LDS loads, then
alternated B2/B3 into the old and new banks. `lgkmcnt(2)` retired the older
two-load fragment while four current-fragment WMMAs covered the newer pair.
The candidate preserved all 16 LDS loads, 16 WMMAs, arithmetic, global refill,
and barriers, and remained full-output exact.

The larger live range declared 128 VGPR and unexpectedly admitted three
blocks/24 waves at the unchanged 18 KiB LDS footprint, but reached only 46.180
TFLOPS. Reserving an otherwise-unused 24 KiB group segment capped the identical
instruction stream at two blocks/16 waves and improved only to 46.470 TFLOPS.
The loss is therefore intrinsic to the deeper B/LDS pipeline rather than
extra-residency contention. This path is closed without qualification. Raw
outputs are `/root/wmma-results/bfrag-pipeline-screen-20260824.txt` (SHA-256
`803b3dac3529abffd0bd9130a08aa93053947514b1a9df67705ed5bb755c3c9a`) and
`/root/wmma-results/bfrag-pipeline-cap2-screen-20260824.txt` (SHA-256
`2a74aa5c40f5b61ed155f8e87ca3ba02feaf12adde3a6850bd98994834581142`).

AMD's 2026-08-06 RDNA 3.5 XML defines `S_CLAUSE` length as
`SIMM16[5:0] + 1`, so the leader's `s_clause 0x1` deliberately gives the two A
global-refill loads uninterrupted service while leaving the third B load
outside. Exact short controls tested a three-load clause, a same-size `s_nop`,
and deletion. They reached 49.545, 49.461, and 49.568 TFLOPS, bracketed by
49.378/49.283 controls, making deletion look attractive.

Six alternating-order 20-warmup/100-iteration pairs rejected that short
signal. Clause-free candidate medians were 48.929, 48.888, 48.953, 49.085,
49.201, and 48.971 TFLOPS (49.004 average); controls were 49.299, 49.116,
48.951, 49.134, 48.985, and 48.944 (49.072 average). The exact candidate lost
0.067 TFLOPS on average. The two-load clause remains selected. Raw outputs are
`/root/wmma-results/refill-clause-screen-20260824.txt` (SHA-256
`51f3c59d6feeaa1da3c4b3aaad2ad05bacdf632e264de1cbdf05619139bdee1b`) and
`/root/wmma-results/refill-clause-qualification-20260824.txt` (SHA-256
`c0d4133bed5a18b95e257515301cab92be0898be097c6523670c4bf693133942`).

Safe VGPR over-reservation explained the otherwise surprising occupancy seen
in the 128-VGPR B-pipeline image. Leader variants declared 121, 124, or 128
VGPR while still using only `v0:v119`; after removing those two metadata fields,
all four assembly sources had the same SHA-256. Every over-reserved image was
exact and changed device-reported occupancy from two blocks/16 waves to three
blocks/24 waves. Throughput fell sharply to 46.035, 46.225, and 45.823 TFLOPS,
between untouched controls at 49.670 and 49.448. The third CU-local
LDS-sharing group is a synchronization/contention regression, not free latency
hiding. Retain the honest 120-VGPR declaration. Raw output is
`/root/wmma-results/vgpr-reservation-screen-20260824.txt` (SHA-256
`6fc1f9b063b515c63aba7440306eaf474b5cdf96a7823f2f66ada6137abb53d0`).

The XML-defined `S_SET_INST_PREFETCH_DISTANCE` was then tested separately from
the earlier loop-alignment work. Modes 1/2/3 request 1/2/3 cache lines ahead
while retaining 2/1/0 lines behind. The instruction was inserted once before
the K-loop label, so the back edge does not re-execute it; a same-size `s_nop`
isolated code placement. All images remained exact with unchanged resources.
Modes 1, 2, and 3 reached 49.149, 49.281, and 49.446 TFLOPS, while the NOP
reached 49.076 between controls at 49.480 and 49.298. No explicit mode beats
the launch default repeatably, so the leader remains unchanged. Raw output is
`/root/wmma-results/inst-prefetch-mode-screen-20260824.txt` (SHA-256
`8e9e835354ec06ff3faf26acdd9c77ee713740991a372a160721192f5f25a18c`).

The separate code-object initial-prefetch field was swept at 0, 8, 12, 16,
and 32 units versus the compiler default 63 (units are 128 bytes). All images
were instruction-identical and exact. The short screen produced
49.584/49.425/49.299/49.547/49.348 TFLOPS between 49.458/49.557 controls, so
size 0 advanced to qualification. Six alternating 20-warmup/100-iteration
pairs averaged 49.0069 TFLOPS for size 0 and 48.9999 for size 63, a negligible
+0.007-TFLOPS difference with four candidate wins. Disabling initial prefetch
is neutral, not a promotion; retain the compiler default. Raw outputs are
`/root/wmma-results/inst-pref-size-screen-20260824.txt` (SHA-256
`cf58d8a255a8cb2188281103253b6818fcc5981ac526efc0d7e2e75e2b32cc89`) and
`/root/wmma-results/inst-pref-size-qualification-20260824.txt` (SHA-256
`b6735eeeba1971e90cee5a2b098a2c5395f871a669ad3ea67023e683bf81d907`).

Periods 2 and 4 were subsequently regenerated through the complete current
hand chain; they had existed only in an older source-level build sweep. Both
were exact at the same 120-VGPR/22-SGPR/18-KiB resource tuple. Period 2 reached
48.534 TFLOPS and period 4 reached 49.375, bracketed by period-16 controls at
49.701 and 49.600. Neither advances, completing the hand-scheduled traversal
set at 2/4/8/16/32. Note that `WMMA_BP_SWIZZLE` controls the workgroup tile
mapping, not LDS address swizzling. Raw output is
`/root/wmma-results/mapping-swizzle-small-screen-20260824.txt` (SHA-256
`92089904bb1ccb3b0160af25651413737fda32d9b100980cd8ddf106b339696e`).

The 4096 grid-tail hypothesis was measured rather than inferred. A 256x128
tile produces 512 workgroups; dividing them over 40 CUs suggests a 12/13-block
tail with a nominal 98.46% utilization ceiling. The harness now supports exact
compile-time rectangular M/N/K dimensions and independently sizes, packs,
references, and validates A, B, and C. Its unchanged 4096 default rebuilt and
reproduced the selected full-output tuple.

Two 480-workgroup shapes divisible by 40 were tested with the identical device
image. The 3840x4096x4096 (15x32 grid) case reached 47.607/47.668 TFLOPS versus
49.443/49.654 square controls. The complementary 4096x3840x4096 (16x30 grid)
case reached 47.280/47.673 versus 49.412/49.388 controls. Every output element
in all eight processes passed the reference. Since removing the nominal tail
reduces normalized throughput in both orientations, simple 512/40 imbalance
does not account for the remaining 50-TFLOPS gap; a fractional-tail kernel is
not justified. Raw outputs are
`/root/wmma-results/tail-grid-diagnostic-20260824.txt` (SHA-256
`991afd2ea95963c04d6ef76c2f1c63eedc2cd49304d979ec4d561bb0e9a2958f`) and
`/root/wmma-results/tail-grid-n-diagnostic-20260824.txt` (SHA-256
`bce6d6eb05fa13bde643f6d5d3f3d86e6abcc5a6b8990050d44ed759e2914f6a`).

The next synchronization experiment partitioned workgroups by
`(blockIdx.x / 40) & 1`, using exact scalar magic-number division in the
otherwise-dead entry lifetime of `s19`. AMD's 2026-08-06 RDNA 3.5 XML confirms
both `S_SLEEP` and `S_SETPRIO`. One-time sleep depths 1/2/4/8/16 and persistent
priority partitions retained the leader's 120-VGPR/22-SGPR/18-KiB resource
tuple and exact output. The promising sleep-1 screen did not qualify: six
order-swapped long pairs averaged 49.058 TFLOPS for the skew and 49.098 for
the leader, with only two candidate wins. Giving priority 1 to the first or
second nominal residency round reached 46.802 and 48.959 TFLOPS against
49.533/49.521 controls. Cross-workgroup phase manipulation is closed; fair
hardware arbitration is better. Raw qualification SHA-256 is
`8389ff4180a78bcf17e3b0a0bc730bdbb587d1da4fd1fe7048833ffcf32f1270`.

Two LDS/output dataflow experiments also remained below the selected image.
First, a new exact pair-lane epilogue used the XML-supported DPP form of
`V_PACK_B32_F16`: odd lanes gather the adjacent even-lane accumulator value
and write an aligned dword. Direct half operands removed all 128 extraction
instructions, but the final 119-VGPR source reached only 46.636 TFLOPS versus
49.300/49.398 controls. Pairing lanes halves active memory transactions, not
the 128 static wave-level store instructions, and adds 128 DPP packs. Raw
SHA-256 is
`41a659e89a94284cfcfe1614c572f27f906850ce38e8c7c7e09a2aaed4f9fd22`.

Second, physical 8- or 16-half gaps between 16-row LDS fragments gave A and B
fragments different bank phases while preserving b128 row transfers. All six
A-only, B-only, and combined layouts were exact. They reached 37.724--47.444
TFLOPS against 48.286/48.427 source controls; B16 was best at 119 VGPR, while
the other layouts needed 133--144 VGPR to retain independent address bases.
The original 768-byte, bank-period-aligned fragment spacing remains selected.
Raw SHA-256 is
`b9c3c86cbaae3fccc4eb6634e9b5a59787650df95181d8839c6bd6f718dd65ff`.

An independent-dispatch architecture was tested after the workgroup-local
phase controls. The selected 512-workgroup image was split into two standalone
HIP code objects; the second differs only by one entry `s_addk_i32` that maps
its local IDs onto the remaining output tiles. Both retain 120 VGPR, 22 SGPR,
18 KiB LDS, and two blocks/16 waves. The opt-in harness launches the images on
separate streams, times both from a common device event, uses the slower stop
as full-output completion, and validates all 16,777,216 results.

The exact 240/272, 256/256, 280/232, and 320/192 splits reached 48.720, 48.633,
47.160, and 47.490 TFLOPS, respectively, between one-dispatch controls at
49.258/48.995. Six alternating long pairs then compared the best 240/272 form:
it averaged 48.629 TFLOPS (48.452--48.889) versus 48.913
(48.736--49.211) for the selected image and lost every pair. A serial 256/256
split reached only 46.379 TFLOPS. Separate queue scheduling cannot repay the
extra dispatch boundary, so the single launch remains selected. Raw screen and
qualification SHA-256 values are
`d0076d87a0ce411d102349f0c5db87b98c217d646ca18e5cc3ccd3ebe2669712`
and `254a503d946b985b626f2b3790929808d99584321f0171f8e211c34c5644df7c`.

The XML-supported VALU/LDS clause mechanism was also screened. A guarded
assembly transform added `S_CLAUSE` only before the selected loop's natural
WMMA runs (2/4/4/3 instructions) or LDS-load runs (10/2/2/2), with identical
placement NOP controls. All forms retained 120 VGPR, 22 SGPR, 18 KiB LDS, two
blocks/16 waves, and exact output. The short WMMA4 and LDS10 forms reached
49.314/49.480 TFLOPS versus 49.016/49.327 NOPs, but the deeper screen rejected
the signal: WMMA4+LDS10, WMMA4, and LDS10 reached 49.222, 49.190, and 49.284
inside selected-image controls at 49.332/48.995. Clausing all natural runs was
worse. Ordinary arbitration remains selected. Raw SHA-256 values are
`5f4d5ef8027e8fdb6b416ab3a9d5c9e70fef8a886c1af90913630403f8b0e940`
and `87531467e660c58ed3c7c816e7ac3e1cbb3e0e9e249e67915c59bbcbaab63f46`.

A new symmetric 192x192 architecture used four-by-three waves, 48x64 work per
wave, exact one-vector-per-thread operand ownership, 115 VGPR, 22 SGPR, and the
same 18 KiB LDS footprint as the leader. The harness now supports honest
zero-padded block contracts for non-divisor tiles: M/N pack to 4224 outside
timing, edge stores remain bounded to 4096, and only logical 4096-cubed work is
reported. Full-output validation reproduced the selected error tuple.

CU mode admitted one block/12 waves and reached 42.757 TFLOPS versus
49.399/49.437 controls. Changing only the code-object placement to WGP exposed
three blocks/36 waves but reached 44.034 versus 49.864/49.530 controls. The
6.35% padded physical work cannot account for the remaining deficit: CU mode
strands four local wave slots and WGP mode pays the known cross-CU handoff
cost. The geometry is closed. Raw SHA-256 values are
`50008ee36fba35af2863f64ff5ee1658d8e30534ccd3573529c4f70015e72cac`
and `8de2600587dd01d44aa336dacd5bf6fa92df72e1a90cd0dabd731c24ac4b5ce7`.

An eight-wave 192x192 follow-up changed each wave from a 48x64 tile to a
48x96 tile (3x6 WMMAs per K16). This gives two clean wave columns and assigns
exactly three b128 input vectors to every thread: one A and one B vector for
all lanes, then an extra A vector in waves 0--3 or B vector in waves 4--7.
Refills were deliberately placed after each A fragment's final WMMA so LLVM
could reuse the dead fragment registers. Inspection of the emitted gfx1151
ISA confirmed the three late `global_load_b128` operations reuse those ranges;
the image uses 122 VGPR, 26 SGPR, 18 KiB LDS, and no scratch. Its WMMA, wait,
and monolithic-barrier sequence was checked against the 2026-08-06 RDNA 3.5
machine-readable ISA XML.

Both placements reproduced the complete selected error tuple and reported
three resident blocks/24 waves, so this was not an occupancy failure. CU mode
reached only 33.229 TFLOPS and WGP mode 37.333, between selected-image controls
at 49.452/49.536. The same 6.35% zero-padding charge applies, but it is far too
small to explain the deficit. Six accumulators per wave enlarge the fragment
working set and the three-vector refill schedule cannot overlap enough useful
work to compensate. The 3x6 ownership is therefore closed without long
qualification. `build-block-192x192-wide.sh` reproduces both placements. Raw
SHA-256 is
`3a92f013b2beb20a1fa96bb0516b9b4c633a8babd134d723954345927ad72e92`.

The selected loop's MUBUF cache policy was then exhaustively screened from
the RDNA 3.5 XML rather than inferred from another generation. `ENC_MUBUF`
defines independent `GLC` (globally coherent), `SLC` (system-level coherent),
and `DLC` (L1 coherent across WGPs in a shader engine) bits. A guarded assembly
transform changed only those bits on the three refill loads; all eight images
kept the leader's instruction count, 120 VGPR, 22 SGPR, 18 KiB LDS, two
blocks/16 waves, and complete exactness tuple.

The unmodified policy won at 49.498/49.579-TFLOPS opening/closing controls.
`GLC`, `SLC`, and `DLC` reached 49.216, 48.914, and 45.538 TFLOPS. The
`GLC+SLC`, `GLC+DLC`, `SLC+DLC`, and all-bit forms fell to 48.308, 46.522,
28.891, and 33.460 TFLOPS. Coherent cache behavior does not improve the
packed-input reuse pattern; the default MUBUF policy remains selected.
`build-cache-policy.sh` and `tools/patch_cache_policy_asm.py` reproduce the
screen. Raw SHA-256 is
`0234fcb18c3d5ebb105909b4974838d9efd663202ef1b3a5a46c41cbb294c7be`.

The earlier invalid 5x8 mapper splice was resolved without transplanting its
incompatible source prologue. On the exact 16x32 grid, the selected 16-wide
XOR-snake mapper has a closed-form inverse. A guarded entry transform now feeds
it `f^-1(g(workgroup_id))`, where `g` is the 5x8 traversal. An offline proof
checks all 512 IDs for bijection and coordinate equality. This changes only
one-time SALU setup and raises the declaration from 22 to 34 SGPR; the entire
hot loop remains text-identical at 120 VGPR, 18 KiB LDS, and two blocks/16
waves.

Both the ordinary and cyclically skewed 5x8 forms reproduced the full output
tuple. They reached 49.242 and 49.067 TFLOPS versus opening/closing selected
controls at 49.487/49.254. Thus a 40-workgroup supertile is now a valid measured
mapping, but it does not improve the selected traversal. The exact inverse-ID
method is retained for future mapping experiments; neither 5x8 form advances.
`build-workgroup-remap-5x8.sh` and `tools/remap_workgroup_5x8_asm.py` reproduce
the result. Raw SHA-256 is
`629424fcbf89ea95514933ec8bf12e87ac78643f1bb097462a8598047af33fad`.

A new ping-pong architecture then used storage the selected layout had left
idle. Each p8 A row contains 16 data halfs and eight padding halfs; across 256
rows, that padding is exactly the 256 b128 half-rows needed by a 128x16 B tile.
The B halves are embedded with
`slot(row,h)=((row+2+3h)&7)+8*(2*(row>>3)+h)`. Exhaustive construction proves
that all 256 slots are unique, and
`slot*48+32 == row*48+h*16 (mod 128)`, so B retains the selected layout's LDS
bank phase. Two combined A+B buffers occupy 24 KiB instead of 36 KiB, keeping
two blocks/16 waves while reducing the hot loop to one monolithic `S_BARRIER`
per K16 publication. AMD's 2026-08-06 RDNA 3.5 XML confirms that B128 is the
widest LDS vector and that only the monolithic workgroup barrier is available.

The first exact implementation used 121 VGPR, 22 SGPR, 24 KiB LDS, and reached
39.291 TFLOPS. Matching the selected progressive A/B wait order raised it to
39.973. The final address transform exploits
`slot(row+16,h)=slot(row,h)+32`: all four B fragments share two lane-dependent
bases and use constant 1,536-byte steps. Emitted ISA contains only those two
dynamic B bases, uses LDS offsets 0/1536/3072/4608 relative to them, and drops
to 117 VGPR without scratch. Full-output validation again produced normalized
maximum error 0.018779343, RMS error 0.035428338, and cosine similarity
0.999977929.

That optimized form reached 41.798 TFLOPS between selected controls at
49.406/49.186. Reclaiming a barrier and 12 KiB of double-buffer storage cannot
repay two noncontiguous LDS reads per B fragment plus their exposed waits. The
embedded-padding pipeline is therefore closed without long qualification;
`build-embedded-ab-pingpong.sh` reproduces it. Raw final-screen SHA-256 is
`d1a9efe242d66568f6c690cb241a8defa2f7f42164acf12a355b9d31d9e076d0`.

The RDNA 3.5 WMMA replication rule also enabled a genuinely different B-load
experiment. Since lanes 16--31 must duplicate lanes 0--15, the lower row can
load the low b128 half while the upper row loads the high half from one
lane-dependent LDS address. Four `V_PERMLANEX16_B32` operations gather the
opposite row's dwords, and four `V_SWAP_B32` operations under the upper-row
EXEC mask normalize the fragment order. The 2026-08-06 XML explicitly defines
the former as a VALU gather across two 16-lane rows and the latter as a
two-VGPR swap.

`tools/patch_halfwave_b_asm.py` applies this only to the 255-iteration steady
state; the final K slice remains byte-for-byte unchanged. It halves B LDS
traffic from eight to four b128 loads per K16 while retaining 120 VGPR, 18 KiB
LDS, no scratch, and two blocks/16 waves; SGPR use rises from 22 to 24 for the
identity lane selectors. The candidate reproduced the complete error tuple but
reached only 46.212 TFLOPS versus 49.677/49.701 controls. Sixteen cross-row
gathers and sixteen swaps per K16 cost more issue/latency than the four saved
LDS instructions. The vectorized in-wave transpose is closed without long
qualification. `build-halfwave-b.sh` reproduces the image; raw SHA-256 is
`d9c5b64f4ed2a7fedc44994ac3dcb1b814b6cf7fe83514242eddcdf991e7be16`.

A one-buffer K32 publication architecture then tested barrier amortization
directly. Each physical LDS row holds two K16 slices plus p8 padding (40
halfs): A occupies 20 KiB and B 10 KiB, so the complete 30-KiB tile preserves
two blocks/16 waves per CU. One read-complete/publication barrier pair now
covers 32 WMMAs, rather than 16. This is the available architectural route on
gfx1151: AMD's 2026-08-06 RDNA 3.5 XML exposes B128 as the widest LDS transfer
and only the monolithic `S_BARRIER`, not a split arrive/wait primitive.

The C++ source exposed a useful compiler boundary. Inline LDS operations made
the intended 32 b128 reads visible, but LLVM allocated 152 VGPR and retained a
large address forest. `tools/patch_k32_publication_asm.py` replaces those
addresses with one A base, one B base, and immediate row/slice offsets; folds
the six refill vectors into dead fragment banks; restores explicit progressive
LDS waits; and emits 120 VGPR, 22 SGPR, 30 KiB LDS, and no scratch. A first
compressed image corrupted the final tile because its compiler-specialized
tail still depended on address VGPRs repurposed by the hot loop. Giving the
tail immutable bases and the same canonical offsets as the repeated body
restored the complete selected error tuple: normalized maximum error
0.018779343, RMS error 0.035428338, and cosine similarity 0.999977929.

The final 10-warmup/5x5 bracket reached **41.439 TFLOPS**, between selected
K16 controls at 49.421 and 49.811 TFLOPS. Halving barrier frequency does not
repay the larger LDS footprint, doubled fragment-read body, and longer live
ranges. The K32 publication architecture is therefore closed without long
qualification. `build-block-k32-publication.sh`,
`block_k32_publication.hpp`, and the guarded assembly transformer reproduce
the result. Raw output is
`/root/wmma-results/k32-publication-selected-screen-20260824.txt` (SHA-256
`abf771ec2a470b55b0b1519054a6b11a89927d9aa8609e21a23abee9028f334e`).

The next prepacking experiment aligned the physical K-tile strides of A and B.
A 256x16 A tile advances by 8 KiB, while the compact 128x16 B tile advances by
4 KiB. Padding each B tile to 8 KiB lets both refill streams use the same
scalar offset. A guarded assembly transform changes the B block-base shift from
12 to 13, retargets the hot B MUBUF load from `s18` to A's `s7`, and removes
the separate B-offset initialization and repeated `s_addk_i32`. The repeated
loop is one SALU instruction shorter, with unchanged WMMAs, memory operations,
barriers, 120 VGPR, 22 SGPR, 18 KiB LDS, and two blocks/16 waves. The cost is a
2x persistent B representation (64 MiB rather than 32 MiB at 4096 squared).

An exact three-warmup smoke process reached 51.182 TFLOPS, but the standard
long qualification rejected it. Six alternating-order 20-warmup/100-iteration
candidate processes measured 48.520, 48.722, 48.936, 49.042, 48.927, and
49.023 TFLOPS (48.862 average). Their paired compact-stride controls averaged
48.847 TFLOPS. The candidate won three of six pairs and its +0.014-TFLOPS
(+0.029%) mean delta is noise, far below the five-process >50 requirement.
Every process reproduced the complete selected error tuple. Shared-stride B
prepacking is retained as a reproducible negative result, not a promotion.
`build-shared-k-stride.sh` and `tools/patch_shared_k_stride_asm.py` reproduce
the image; raw output is
`/root/wmma-results/shared-k-stride-qualification-20260824.txt` (SHA-256
`5a318fe0060d29551d564b0ef621f8d34a4e0a848b932dd000ad7ae708a07e9a`).

The padding-free compact-XOR experiment then tested a true 24-KiB A/B
ping-pong stage. Its bijection
`slot(row,half)=2*row+(half^((row>>2)&1))` distributes each WMMA half evenly
over all eight 16-byte LDS phases and cuts publication to one monolithic
barrier per K16. The original flat-load source image was exact at 185 VGPR.
One-, two-, and
three-bank just-in-time B recycling was not reliable: shallow K diagnostics
passed, repeated K=128 processes became nondeterministic, and full-depth error
was about 0.38 normalized. The 2026-08-06 XML also corrected the diagnostic
interpretation: `0xfe9f` selects `VA_SSRC=0`; RDNA 3.5 has no `VA_VSRC`
dependency field.

Keeping four independent B fragments restored the complete exactness tuple.
The first retained hand image overwrote each B bank with a refill vector only
after that fragment's final WMMA, removing dedicated refill registers without
reusing a fragment in phase. It emitted 153 VGPR, 22 SGPR, 24 KiB LDS, no
scratch, and two blocks/16 waves. A same-pass 10-warmup/5x10 bracket measured
43.758 TFLOPS between selected controls at 49.702 and 49.816 TFLOPS.

A follow-up moved address formation to compiler-created raw-buffer resources
using `__builtin_amdgcn_make_buffer_rsrc`. That detail is mandatory on this
host: hand-built descriptors made from the flat pointer SGPRs lost AMDGPU
aperture information and faulted, while the compiler descriptors were exact.
The source MUBUF image uses 180 VGPR and reaches one block/eight waves. The
hand schedule preserves those descriptors, issues each refill only after its
B fragment dies, and reduces the image to 145 VGPR and two blocks/16 waves.
It passed K=32 and full-K exactness. The 10-warmup/5x10 bracket measured
**45.945 TFLOPS** between selected controls at 49.887 and 49.790 TFLOPS.

Static audit also corrected the earlier diagnosis: compact-XOR and the leader
both issue 16 b128 fragment LDS reads per K16, not a 2x difference. The
remaining loss is in the front-loaded LDS dependency schedule and refill /
publication critical path; saving one barrier still does not compensate. The
architecture remains a reproducible negative result and is not qualified.
`build-compact-xor-pingpong.sh` reproduces it; raw output is
`/root/wmma-results/compact-xor-mubuf-hand-bracket-20260825.txt`
(SHA-256
`c5108455a596b8e011ff5d3794941b25bed435a48bf9cc20f7eb5d896599495b`).

At that point none of these results changed the qualified 49.143-TFLOPS
research leader or the requirement for five fresh exact process medians above
50 TFLOPS. The register-placement result below subsequently cleared that gate.

### 2026-08-25 independent hot-B register phase clears 50 TFLOPS

The selected delta-2 allocation established that physical VGPR placement is
load-bearing, but earlier +4/+6 boundary shifts also changed the declared
allocation and runtime residency. `tools/rotate_hot_fragment_phase_asm.py`
isolated all eight modulo-eight fragment phases while retaining 120 VGPR,
22 SGPR, 18 KiB LDS, no scratch, and two blocks/16 waves. Every phase was
exact. Phases 1/2/5/6 lost about 2%; phases 0/3/4/7 stayed near the selected
phase-4 control. No uniform rotation improved the leader.

The follow-up kept all four A fragments on their proven phase-4 banks and
placed the repeatedly loaded B fragment independently. `v108:v115` (phase 4)
and `v112:v119` (phase 0) remained below 50, while `v111:v118` (phase 7)
reached 50.530 TFLOPS in the short exact screen between 49.552/49.524 controls.
The transform rotates A2/A3 into the vacated B slots and stages the B refill in
`v72:v75`; it changes no repeated opcode/order, wait threshold, memory address,
WMMA count, or memory-operation count and adds only two one-time address-shadow
moves before the K loop.

Five fresh processes then used 20 warmups and five 100-iteration timing blocks
with candidate/control order alternated. Candidate medians were **50.014850,
50.135584, 50.146413, 50.025904, and 50.046643 TFLOPS**: **50.073879 TFLOPS
average**, 50.014850 floor, and 2.744730-ms average median time. Paired controls
averaged 49.018881 TFLOPS. All ten processes reproduced normalized maximum
error 0.018779343, RMS 0.035428338, and cosine similarity 0.999977929 over all
16,777,216 outputs. This is a +2.152% same-pass uplift and clears the promotion
rule without changing the numerical or persistent-input contract.

`build-hot-fragment-phase.sh`, `build-hot-b-phase.sh`,
`tools/rotate_hot_fragment_phase_asm.py`, and
`tools/place_hot_b_phase_asm.py` reproduce the sweep and selected image. Raw
qualification output is checked in at
[`results/hot-b-phase111-qualification-20260825.txt`](../results/hot-b-phase111-qualification-20260825.txt)
(SHA-256
`16cd5054723ec6db71b61ce27b0fe9d7d23f009471b9297d31e815090927c6b7`).
See [Reproducing the 50 TFLOPS qualification](REPRODUCING_50_TFLOPS.md) for
the pinned toolchain, literal build and five-process commands, pair order, and
acceptance criteria.
