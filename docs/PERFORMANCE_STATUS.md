# gfx1151 WMMA performance status

This is the current performance ledger and optimization plan for the repository.
It was refreshed on 2026-08-23 from source base `8f7d926` plus the standalone
record harness added with this documentation update. In-progress extension
kernel changes in the shared worktree were deliberately not included. The older
development notes remain useful as a lab notebook, but their claims are not
automatically current.

## What the repository has actually measured

The current validated peak is **41.322 TFLOPS** (3.326 ms) for a
4096 x 4096 x 4096 GEMM with FP16 inputs and FP32 accumulation/output. It uses
the pinned standalone harness in `tools/`, deterministic random inputs, and a
full rocBLAS FP32 reference over all 16,777,216 outputs. It exceeds the
historical 41 TFLOPS target, but it has not passed the stricter sustained
promotion gate: five fresh 100-iteration processes had a 40.900 TFLOPS median,
a 40.772-41.104 range, and two passes above 41.

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

### Unmeasured large-tile candidate

`matmul_opt` is bound in Python but is absent from the benchmark lists. Its
actual kernel is a 128 x 128 work-group tile with eight waves and a 2 x 4 WMMA
tile per wave (eight FP32 accumulators). The binding text and wrapper comments
that describe a 256 x 128 tile, 4 x 4 warp tiling, or a vectorized epilogue are
stale.

This is the best first experiment because it doubles useful output work per
barrier and halves staged A traffic per output element relative to the record
kernel. It also carries substantial risk: eight accumulator fragments consume
64 VGPRs per lane before A/B fragments, transpose temporaries, addresses, and
loop state are counted. A spill or occupancy collapse can erase the reuse gain.

### Historical names

`matmul_zerocopy` performs vector global loads for B, followed by eight scalar
stores into transposed LDS, and uses the same scalar direct-to-global epilogue
as the standard kernel. "Zero-copy" and "swizzled B" are historical names, not
descriptions of the current data path.

## Current plan to sustain more than 41 TFLOPS

The peak target has been crossed. The remaining work is to make that crossing
durable and integrate the schedule without weakening the numerical contract.

1. **Reduce the fresh-process floor by at least 0.56%.** The current minimum is
   40.772 TFLOPS. Optimize against the worst and median process, not the best
   short block.
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
5. **Promote only after five fresh processes.** The full FP32 reference must
   pass in every process and every process median must exceed 41 TFLOPS.

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
   22.03 TFLOPS under a 2% noise margin, but the active project gate is stricter:
   every one of the five fresh-process medians must exceed 41.0 TFLOPS.

Current ROCm documentation lists gfx1151 as supported by ROCm Compute Profiler.
Counter availability still depends on the installed driver and profiler, so
query the target host and document unavailable counters instead of assuming
that all counters work or that none do.
