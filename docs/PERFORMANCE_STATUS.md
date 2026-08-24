# gfx1151 WMMA performance status

This is the current performance ledger and optimization plan for the repository.
It was refreshed on 2026-08-24 with the ordinary-layout standalone harness,
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
research leader at 49.035 TFLOPS; it has its own 50-TFLOPS promotion gate.

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

The current FP16-output, block/K16-prepacked research leader averages **49.035
TFLOPS** at 4096 cubed across five fresh 100-iteration processes. Their medians
are 49.095/48.980/48.995/48.986/49.116 TFLOPS, with a 48.980-TFLOPS floor and
2.802906-ms average median time. It uses a 256x128 block, eight waves, p8 A/B
LDS rows, 120 VGPR, 22 SGPR, 18 KiB LDS, and no spills. The result passed a
full rocBLAS reference check in every process, but it is a persistent-input
contract: packing is outside the timed region. The previous sustained leader
was 48.614 TFLOPS. A 49.573-TFLOPS short sample and two isolated 50+ samples
remain non-promotable; the sustained 50-TFLOPS FP16 gate is still open.

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

## Current plan to sustain more than 50 TFLOPS

The ordinary-layout peak has been crossed and the prepacked contract is within
1.97% of the target. The remaining work is to make a 50-TFLOPS crossing durable
without weakening either numerical contract.

1. **Close the prepacked sustained gap.** The current five-process floor is
   48.980 TFLOPS and the average is 49.035; optimize against the worst and
   median fresh process, not the best short block.
2. **Attack the measured synchronization cost.** The one-buffer K32 path uses
   two workgroup barriers per step. Explore a split signal/wait schedule or a
   shared-memory layout that permits overlap while preserving the current
   two-workgroup LDS residency. Reject any design that increases scratch or
   loses full-output correctness.
3. **Separate package behavior from kernel behavior.** The 1,000-warm-up test
   slowed over sustained execution. Capture clocks and package power in a
   separate diagnostic pass, then keep the official timing pass free of
   counters and polling.
4. **Port the exact schedule into the PyTorch extension.** Preserve the pinned
   standalone harness as the baseline, then compare the integrated kernel with
   identical preallocated output and inputs before measuring API overhead.
5. **Promote only after five fresh processes.** The full reference must pass in
   every process and every process median must exceed 50.0 TFLOPS for the
   prepacked contract. Keep the ordinary-layout contract's separate gate.

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

The source K2 specialization (`WMMA_BP_K_SLICES=2`) was built for the
256x128 packed record shape to publish two WMMA K16 slices per stage. It
launched at 44.350 TFLOPS but failed the numerical gate badly (normalized
maximum error 1.356285863, cosine -0.000246312), indicating an incompatible
packing/accumulator contract in the generic K2 path. The dedicated K2 ring
variant repaired exactness, but reached only 41.345 TFLOPS. K32 staging does
not expose a route toward 50 TFLOPS without a new packing implementation.

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

### 2026-08-24 asymmetric LDS padding screen

Complementary A/B stride pairs were tested on the packed 256x128 source
kernel: (4,12), (12,4), (2,14), (14,2), (6,10), and (10,6). Every candidate
passed the exactness tuple and retained two-block occupancy, but the
20-warmup/20-iteration screens reached only 25.755--27.389 TFLOPS. The
operand stride phases are coupled by the WMMA/LDS access pattern; asymmetric
padding is closed as a route to the leader.
