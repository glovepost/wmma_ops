# Remaining WMMA frontier exploration — 2026-08-23

Target: gfx1151, ROCm 7.14, 4096^3 GEMM, FP16 A/B and FP16 C,
A column-major, B/C row-major unless explicitly labelled prepacked.  Every
reported candidate was checked against a rocBLAS FP32-compute/FP16-output full
reference and passed the promotion correctness gate (finite output and cosine
similarity greater than 0.995).  The common observed cosine was 0.999977929.

The short screen used separate unprofiled timing passes, five timing blocks,
and the shared GPU lock with production stopped/restored.  These are screening
results, not five-fresh-process promotion results.  The existing fresh-process
baseline remains 46.082 TFLOPS (2.982 ms); 50 TFLOPS requires 2.748779 ms.

## Control

| Candidate | TFLOPS | Median ms | Resources |
|---|---:|---:|---|
| Packed-C single-buffer control | 45.581 | 3.015286 | 124 VGPR, 38 SGPR, 12 KiB LDS |

## Inter-wave producer/consumer

| Candidate | TFLOPS | Median ms | Notes |
|---|---:|---:|---|
| 1 producer + 4 consumers, atomic counters | 6.841 | 20.089870 | Correct; 169 VGPR |
| 1 producer + 4 consumers, generation flags | 5.747 | 23.915524 | Lock-free single-writer flags |
| 1 producer + 8 light consumers | 6.368 | 21.583158 | 90 VGPR; 9 waves force one block |
| 1 producer + 4 light consumers, 128x64 | 7.098 | 19.361921 | Designed for three resident blocks |

The decisive limitation is that producer and consumer paths are in one kernel,
so all waves receive the kernel-wide register allocation.  The lower-register
consumer variants recover occupancy only by cutting independent accumulator
chains from eight to four; WMMA latency and the per-K handoff then dominate.
gfx1151 has no split block barrier that would provide a cheaper asynchronous
publish/consume primitive.

## Vectorized in-wave transpose

| Candidate | TFLOPS | Median ms | Resources / lane network |
|---|---:|---:|---|
| 16x16 shuffle butterfly | 10.268 | 13.385176 | Correct after bit-preserving fragment copy |
| 16x16 DPP row-xmask | 12.109 | 11.350111 | 165 VGPR, 512 static DPP moves |
| two 8x8 DPP transposes | 14.278 | 9.625888 | 161 VGPR, 384 static DPP moves |
| four 4x4 DPP transposes | 16.013 | 8.582781 | Correct |
| eight 2x2 DPP transposes | 18.099 | 7.593699 | 141 VGPR, 128 static DPP moves |
| 2x2 DPP on A only | 24.955 | 5.507368 | Original B scalar LDS loads |
| 2x2 DPP on B only | 25.545 | 5.380342 | Best transpose variant |

Smaller butterflies monotonically reduce register pressure and DPP work, but
even the best one-sided form is 44% below the control.  The scalar strided LDS
pattern benefits from bank distribution and replicated-lane multicast; explicit
transpose spends more VALU/DPP work than it removes from the LGKM path.

## Prepacked inputs (separate benchmark contract)

Prepacking occurs before warm-up and is excluded from GEMM timing.  These
results are therefore not record-comparable to the original input contract.

| Candidate | Best TFLOPS | Median ms | Notes |
|---|---:|---:|---|
| Both inputs native, LDS staged | 9.139 | 15.038923 | Best at padding 14 |
| A-only prepacked | 21.599 | 6.363255 | Best at padding 10 |
| B-only prepacked (persistent-weight case) | 20.968 | 6.554676 | Best at padding 14 |
| Both inputs native, direct global fragments | 4.535 | 30.304968 | Zero LDS, duplicated per-wave VMEM |

Full native-padding sweep (TFLOPS): both inputs p2/p6/p10/p14 =
8.175/6.398/5.644/9.139; A-only = 21.445/21.588/21.599/21.483;
B-only = 19.291/20.189/19.166/20.968.  Native fragments turn each lane into a
wide LDS reader from a different row, producing bank conflicts.  Direct global
fragments avoid LDS but multiply operand traffic across consumer waves.

## Follow-on accumulator and scheduling architectures

The promotion gate was tightened after the structured-buffer experiments to
include the baseline's normalized maximum error: finite output, cosine above
0.995, and normalized maximum error below 0.03.  The accepted candidates below
all reproduce the baseline's 0.018779343 normalized maximum error.

| Candidate | Best TFLOPS | Median ms | Notes |
|---|---:|---:|---|
| Fully unpacked 4x4 accumulators | 41.452 | 3.3156 | 8 independent FP16 chains, 2 K slices |
| Unpacked rectangular/staircase tiles | 41.396 | 3.3200 | Best 7x2 logical accumulator grid |
| 4-wave shared staging | 42.983 | 3.1977 | Confirms reuse beats reduced synchronization |
| Structured add-TID SRD, A | 45.570 | 3.0163 | Correct but neutral |
| Structured add-TID SRD, B | rejected | — | Wrong vector semantics; normalized max error 0.1001 |
| Alternate Morton/row/column block mappings | 45.350 | 3.0300 | Existing XOR snake remains best |
| Selectively unpack one N column | 45.238 | 3.0381 | 8 extra VGPRs |
| Selectively unpack two N columns | 45.914 | 2.9934 | Noise-band best; no promotion |
| Selectively unpack all N columns | 45.277 | 3.0355 | Full false-dependency removal did not help |

The structured-buffer B failure was resolved with a lane-level microtest: one
128-bit structured load with add-TID returns indices 0,1,32,33,64,65,96,97 for
lane zero (and 8,9,40,41,... for lane one), not eight contiguous halves.  It
would need four loads and four LDS stores, so that route is closed.

An N-packed/N-major accumulator architecture is compiled and awaiting a GPU
screen.  It keeps eight physical accumulators while pairing logical N tiles and
placing low/high OPSEL updates eight WMMA issues apart.  Its static code object
uses 116 VGPR versus 124 for the M-packed control, with the same 32 WMMAs, 256
LDS loads, four synchronization points, and 12 KiB LDS.  Padding, 64/256-bit
loads, swizzle, eager-load, two-K-slice, and 128x256 block variants are ready.

A second-generation producer/consumer kernel is also compiled: two or four
producer waves partition global-to-LDS traffic and publish each ring slot only
after a per-slot completion epoch.  The 2-producer/8-consumer code remains at
90 VGPR on the consumer path.  Runtime screening is pending because another
shared-box job currently holds the advisory GPU lock while waiting for the
supervised production container to disappear before invoking the production
stop wrapper; that ordering is self-deadlocked.

### Block/K-major persistent-input architecture

The earlier whole-matrix prepack lost global coalescing.  A new contract packs
both operands into block/K-major, 16-wide microtiles before timing.  Each
workgroup retains contiguous cooperative b128 global loads, but stages rows in
the native WMMA fragment order.  The static compute body changes from 128
scalar LDS half-loads and 52--59 waits to 32 `ds_load_b128` instructions, zero
scalar LDS half-loads, and 18 waits.  The 256x128 single-buffer variants use
118--120 VGPR and 12--23 KiB LDS; padding 4 and 8 have the best analytic bank
distribution.  M-packed and N-packed forms are compiled for:

- 256x128, eight waves, six static cooperative global b128 loads;
- 128x256, eight waves, the transposed reuse direction;
- 256x256, sixteen waves, four static cooperative global b128 loads;
- 256x128 ping-pong LDS, one publish barrier per K step (140 VGPR and
  24--36 KiB LDS after compiler scheduling).

These are explicitly a persistent-prepacked input contract, not comparable to
the original-layout record without that qualification.  Full rocBLAS
correctness and timing screens are compiled but pending the same external GPU
lock release.

### Block/K-major measured results after lock recovery

The 256x128 N-packed, padding-8 form is the new correct leader for the
persistent-prepacked contract.  Its steady K16 loop has 16 WMMA instructions,
16 `ds_load_b128`, three cooperative `global_load_b128`, three
`ds_store_b128`, nine waitcnts, and two barriers.  Metadata is 118 VGPR,
22 SGPR, and 18 KiB LDS.  It reproduces the tightened baseline error tuple:
normalized maximum error 0.018779343, RMS 0.035428338, and cosine
0.999977929.

| Candidate | TFLOPS | Median ms | Decision |
|---|---:|---:|---|
| Same-pass original-layout control | 45.697 | 3.007650 | Control |
| 256x128 block-packed M, p8 | 48.340 | 2.843000 | Retained |
| 256x128 block-packed N, p8 | 48.614 | 2.827130 | Previous sustained leader |
| N-p8 + combined schedule + VGPR delta 2 | 49.035 average | 2.802906 average | Five-process sustained leader |
| Best short block-packed pass | 49.573 | 2.772440 | Below 50; not sustained |
| 128x256 transposed-reuse N, p8 | 46.839 | 2.934000 | Rejected |
| 256x256, p8 | 37.680 | 3.648000 | Rejected |
| Four-wave 128x128, p8 | 42.170 | 3.259198 | Rejected; 153 VGPR |

Padding A and B by eight halves is jointly necessary.  Removing either side's
padding falls to 44.85--45.71 TFLOPS.  XOR-snake swizzle 16 remains best;
Morton, row, column, swizzles 2/4/8/32, wait placement, B-fragment pairing,
wave bounds, and separate/combined allocation offsets all stayed within noise
or regressed.  The best offset screen was 48.539 TFLOPS against its 48.401
control.  Loop unroll 2/4 raised VGPR 118 -> 122 and fell to
45.723/43.936 TFLOPS.

Larger K32 staging reduced barrier frequency but raised metadata to 189 VGPR
and reached only 45.588 TFLOPS.  Compiler-generated ping-pong forms use
140--203 VGPR depending on formulation and padding; the measured original
double buffer tops out at 32.43 TFLOPS.  Static pointer alternation, indexed
LDS buffers, and late refill did not clear the resource gate (146, 146, and
159 VGPR respectively at p8), so they were rejected before timing.

Rocprofiler-Compute counters were collected separately from timing at
`/root/wmma-results/profile-blockpack-p8`.  For the block-packed kernel,
`SQ_WAIT_BARRIER / SQ_WAVE_CYCLES` is 14.6% and
`SQ_WAIT_CNT_ANY / SQ_WAVE_CYCLES` is 11.7%; barrier wait alone is large enough
to explain the remaining 2.6% target gap.  This makes a one-barrier K step the
right architectural goal, but only if it preserves the single-buffer kernel's
occupancy.

AMDGPU scheduler bias 0/50/100, max-ILP/max-occupancy/iterative strategies,
WMMA-vnop hoisting, and rewrite-stage toggles all emitted byte-identical kernel
text.  The target register-pressure tracker emitted one real change: it delayed
the next-B global load by one WMMA and reused dead fragment registers, but two
longer A/B pairs regressed from 48.440/48.254 to 47.951/47.559 TFLOPS.

A hand-scheduled fixed-phase ping-pong was then assembled and linked back into
the full validation harness.  It retained 118 VGPR and one barrier per K16
phase, unlike the 140--203 VGPR compiler forms.  An asymmetric p8/p0 layout fit
two 18+12 KiB buffers in 30 KiB LDS and passed the exact full-reference error
tuple after correcting the LDS wait ordering, but reached only 44.368 TFLOPS.
The compact phase's bank behavior and address arithmetic outweighed the saved
barrier.  A symmetric p8/p8 sibling retains 118 VGPR with 36 KiB LDS and is
assembled; it passed correctness but reached only 34.606 TFLOPS.  The compact
p1 sibling reached 22.096 TFLOPS.  Pairing LDS reads at p4 reached 45.088
TFLOPS, while the hand-assembled `ds_load_2addr_b64` version reached 47.382
TFLOPS; neither improved on the retained p8 leader.

### Occupancy, layout, and instruction-schedule follow-ups

The p8 leader reports two active blocks per CU (16 active waves per CU).  A
single short sample reached 50.216 TFLOPS at 2.736961 ms, but the 20-warmup /
20-iteration sustained run reached only 47.944 TFLOPS at 2.866681 ms.  It is a
noise/power excursion, not a promotable result.  A 16-wave 256x128 geometry
reduced allocation to 91 VGPR and admitted 32 waves per CU, but its smaller
per-wave accumulator tile could not feed WMMA and reached only 39.862 TFLOPS.

A hand-edited progressive LDS schedule moved the first B reads earlier and
staged wait thresholds at 6/4/2/0.  Its short screen reached 50.338 TFLOPS at
2.730310 ms, but a same-pass long comparison was a regression: 47.447 versus
47.719 TFLOPS for the unmodified control.  This second isolated 50+ sample is
also rejected.

The compact half-XOR layout suggested by symbolic layout research used 12 KiB
LDS but reached 39.986 TFLOPS as C++ and 46.956 TFLOPS after hand register
commoning.  Padding remains substantially better than XOR remapping for this
gfx1151 access pattern.

### PaperClip literature pass

The research workflow used `matsjfunke/paperclip` at commit `0e05a40` to fetch
and extract current arXiv papers, including Tawa (2510.14719), HipKittens
(2511.08083), the warp-specialization performance model (2506.11209), Kerncap
(2605.03208), LP-GEMM (2604.04599), Hexcute (2504.16214), Harness Engineering
(2607.17979), GPU Forecasters (2605.31464), and Stream-K++ (2408.11417).

The portable conclusion is to preserve consumer occupancy and use a large
output tile with an in-wave pipeline.  Tawa's main mechanisms depend on Hopper
TMA, asynchronous WGMMA, dynamic register reallocation, and mbarriers.
HipKittens independently reports that producer specialization loses on AMD
because registers are statically allocated.  Its direct global-to-LDS
intrinsic is CDNA-only: a gfx1151 microtest with
`llvm.amdgcn.raw.buffer.load.lds` fails instruction selection, closing that
route on this target.  Hexcute motivated the measured XOR layout above;
Stream-K's tail-balancing ideas are less relevant to this exactly divisible
4096-cube workload and would add reduction traffic.

### K32 slice-major two-slot ring

A new persistent contract packs K32 blocks as
`[block][K32][slice][row][K16]`.  Both slices therefore retain coalesced global
loads while sharing one padded K32 LDS tile.  A consumed K16 slot is refilled
in 12 VGPRs, and the next barrier both publishes that refill and retires the
other slot.  The design uses one barrier per K16, 30 KiB LDS, and still permits
two blocks per CU.

The first generic C++ implementation passed the exact full-reference error
tuple but inflated to 202 VGPR, one block per CU, and only 39.292 TFLOPS.  A
minimal dedicated kernel reduced that to 151 VGPR.  ISA inspection showed LLVM
had hoisted all four B LDS fragments, extending 24 unnecessary VGPRs.  The
hand-scheduled form reuses one eight-register B fragment, assembles at exactly
128 VGPR with zero spills and 30,720 bytes of LDS.  Its first exclusive screen
failed because B fragments were loaded after refill stores had started
overwriting the consumed slot.  Moving those loads ahead of the stores
restored the exact full-reference tuple.  Interleaving refill VMEM between the
four B groups improved the correct form from 36.159 to 41.988 TFLOPS sustained
(43.667 TFLOPS short), still behind a same-pass 47.767 TFLOPS p8 control.
Holding two B fragments while recomputing addresses retained the 128-VGPR
allocation but regressed to 39.921 TFLOPS sustained.  The corrected ring is
therefore retained as a negative architecture result, not a candidate.

### Post-sweep research queue

AMD's `tritonBLAS` paper (arXiv 2512.04226) models GEMM locality at the
instruction, register, workgroup, cache, and global levels.  Its cache-scope
factorization gives a concrete gfx1151 mapping experiment: 40 CUs factor as
5x8.  For the retained 256x128 tile, a 5-M by 8-N supertile reuses the larger A
edge across eight workgroups and the B edge across five.  The paper's simple
reuse estimate is `1 - U/R = 85%`, versus 82.5% for the transposed 8x5 choice.
The 4096-cube grid is 16x32 tiles, so it contains twelve exact 40-workgroup
groups plus a 32-workgroup tail.  Mapping modes 5 and 6 implement the plain and
cyclic-row-skewed 5x8 forms; a GPU-free bijection check covers all 512 tiles.
Both compile at 119 VGPR, 26 SGPR, and 18 KiB LDS, so their one-VGPR increase
over the p8 control does not change its two-block/16-wave occupancy class.
Because the control admits two blocks per CU, consecutive 5x8 groups also make
the first 80 resident workgroups an effective 5x16 region that reuses the same
five A tiles across both block slots.

The current Origami implementation in `ROCm/rocm-libraries` commit
`dab5e862a64f05b4f7323886465eb957444573d9` explicitly supports gfx1151, but
its workgroup selector short-circuits to WGM=1 when `NUM_XCD == 1`.  It does
not directly predict 5x8 for Strix Halo.  This candidate is therefore an
extrapolation of the paper's cache-cost model, not a claim about AMD's selected
configuration.  Of the grid-compatible 40-workgroup rectangles, 5x8 and 4x10
have the same estimated unique-input cost; 5x8 leaves one contiguous 32-tile M
row and lets the snake traversal join that tail at the same N edge.

AMD FlyDSL at commit `11c4174d82b7491c2d08d5828a254183f2a8b959` independently
uses a 128x128x32, four-wave, double-buffered gfx11 WMMA kernel.  It confirms
three relevant design choices: 128-bit cooperative copies, eight-row L2
grouping, and explicit VMEM/DS-read/WMMA/DS-write schedule groups.  Its source
also records that the gfx11 v16 WMMA ABI duplicates operands across wave halves
and proposes `ds_swizzle_b32` XOR16 broadcasts to halve LDS reads.  That
broadcast is lower priority here: each 16-half fragment would trade two
half-wave LDS reads for eight cross-lane DS operations, and the retained
kernel's identical upper/lower addresses already benefit from LDS multicast.

Composable Kernel at commit `07944e928fa3d8ec4c60e7d1ba9af043f52be02f`
provides a more directly testable gfx11 scheduling detail.  Its WMMA
"interwave" v1 pipeline phase-aligns workgroup waves, raises `s_setprio` during
the MAC cluster, and lowers it before the LDS handoff so lagging waves are less
likely to extend barrier tails.  Our K16 loop is already synchronized at every
cluster boundary, so `WMMA_BP_SET_PRIO=1` isolates the remaining priority hint
without changing dataflow, allocation, or correctness semantics.
Offline ROCm 7.14 codegen keeps the control's 118 VGPR, 22 SGPR, and 18 KiB
LDS; the final ISA contains exactly one priority-1/priority-0 pair around each
16-WMMA hot-loop and tail cluster.

The 2026 FIBER paper (arXiv 2608.19628) reinforces that static private-register
allocation is the fundamental obstacle to producer/consumer specialization,
but its solution requires new shared-register hardware and ISA support.  It
does not supply an implementable gfx1151 path.

### Paperclip refresh: AMD scheduling evidence and current compiler work

The later Paperclip pass searched arXiv through 2026-08-23 and retrieved the
full text of six relevant papers. [HipKittens](https://arxiv.org/abs/2511.08083)
reports two AMD-friendly schedules: eight-wave ping-pong and four-wave
fine-grained interleave. Its key portability warning is that AMD statically
allocates registers to every wave, so NVIDIA-style dedicated producer waves
can consume registers without contributing output. That matches our measured
producer/consumer regressions and keeps the all-wave p8 kernel as the correct
base. HipKittens also uses `s_setprio` around matrix clusters; our earlier
gfx1151 screen reached only 42.439 versus 47.937 TFLOPS and closes that hint.

[Twill](https://arxiv.org/abs/2512.18134) formulates software pipelining and
warp specialization as a resource-constrained schedule search, while
[Tawa](https://arxiv.org/abs/2510.14719) expresses producer/consumer channels
as asynchronous references. Their mbarrier/TMA/WGMMA mechanisms are NVIDIA
features absent from gfx1151, but the transferable rule is useful: model the
whole dependency graph and live-resource footprint, not an isolated load
latency. [FIBER](https://arxiv.org/abs/2608.19628) proposes new shared-register
hardware and is therefore a hardware research direction rather than a current
kernel implementation.

[Bringing Auto-tuning to HIP](https://arxiv.org/abs/2407.11488) reports that
AMD tuning spaces can have much sharper optima than NVIDIA spaces. That is the
reason for the new exact row-order search rather than trusting one promising
sample. [tritonBLAS](https://arxiv.org/abs/2512.04226) supplies a useful
hierarchical tile/cache model, but the 4096 square already launches 2048
workgroups; Stream-K-style tail balancing is not expected to close this
compute-kernel gap.

### WMMA row-order screen

The delta-2 leader's first four WMMA issues are constrained by progressive
`lgkmcnt(4/2/0)` readiness. The later three four-row fragments are fully ready,
so `tools/reorder_hot_wmma_rows_asm.py` permutes only their independent issue
order. Row 0 remains first in the final fragment because the next global refill
overwrites its A registers. The transform preserves instruction count,
dependencies, LDS layout, and resources; every candidate passed the full
rocBLAS tuple at 120 VGPR, 22 SGPR, 18 KiB LDS, and zero spills.

The first screen ran in a lower package state and is explicitly provisional:

| Row order | TFLOPS |
|---|---:|
| Controls | 44.954 / 44.808 |
| 0132 | 44.853 |
| 0213 | 45.231 |
| 0231 | 44.855 |
| 0312 | **45.348** |
| 0321 | 45.274 |

0312 is +1.04% over that control midpoint, but its absolute value cannot be
compared with the 49.035 qualification. The isolated group screen is prepared
by `build-wmma-row-order-isolation.sh`; re-bracket it only after the Ember
release and under the shared lock.

### Row-order isolation result

The follow-up bracket ran after the release handoff with identical 20-warmup,
five-block, 20-iteration timing settings and the complete correctness check.
It used two controls to expose package drift:

| Schedule | TFLOPS |
|---|---:|
| Opening control | 49.295 |
| g1 = 0312 | 49.285 |
| g2 = 0312 | 49.329 |
| g3 = 0312 | 49.078 |
| g1 + g2 | 49.006 |
| g1 + g3 | 49.091 |
| g2 + g3 | 49.095 |
| all groups = 0312 | **49.401** |
| Closing control | 49.176 |

Every form remained at 120 VGPR, 22 SGPR, 18 KiB LDS, zero spills, and the
full rocBLAS tuple. The all-group form is about 0.34% above the two-control
midpoint, but the sign is not isolated from package/order drift and no fresh
process passed the 50-TFLOPS gate. Keep it as the next short-screen candidate;
the delta-2 allocation remains the qualified research base.

The five-fresh-process qualification rejected it decisively. Each process used
20 warmups and five 100-iteration timing blocks:

| Process | Delta-2 base | All-group 0312 |
|---:|---:|---:|
| 1 | 49.173 | 49.066 |
| 2 | 49.162 | 48.809 |
| 3 | 48.910 | 48.515 |
| 4 | 48.798 | 48.834 |
| 5 | 48.694 | 48.830 |

Every process reproduced the full error tuple, but the candidate lost its
immediately preceding control in all five pairs. This closes row-order
permutations as a scheduling route; retain the unpermuted delta-2 allocation.

### Late one-barrier and geometry sweep

The remaining one-barrier layouts were implemented and screened against a
bracketed p8 leader.  Every accepted timing below passed the same full rocBLAS
reference tuple (normalized maximum error 0.018779343, RMS 0.035428338, cosine
0.999977929).

| Candidate | Resources | TFLOPS | Decision |
|---|---:|---:|---|
| Direct A / shared B | 140 VGPR, 12 KiB LDS | 26.790 | Duplicate direct-A traffic dominates |
| Persistent 80-workgroup A-row owner | 256 VGPR, 18 KiB LDS | 35.709 | One active block; serial N shards lose |
| Interleaved ping-pong, compiler | 186 VGPR, 30 KiB LDS | 40.757 | Correct layout, excessive allocation |
| Interleaved ping-pong, hand p40/o16 | 118 VGPR, 30 KiB LDS | 46.024 | Best compact one-barrier form; behind p8 |
| Periodic compact A / p8 B ping-pong | 118 VGPR, 32 KiB LDS | 44.947 | Two blocks retained; LDS schedule still loses |
| 512x128, selective B loaders | 137 VGPR, 30 KiB LDS | 37.653 | One 16-wave block per CU |
| 512x128, duplicate B loaders | 124 VGPR, 30 KiB LDS | 35.850 | Lower VGPR does not lift the 16-wave CU cap |

The interleaved layout packs two 16-half buffers into each 40-half LDS row.
Its 80-byte row displacement is the modular inverse of p8's 48-byte
displacement and fits two complete operand buffers in 30 KiB.  Hand scheduling
recovered the leader's 118-VGPR allocation and improved the compiler form by
13%, but the altered bank phase still costs more than the removed barrier
saves.  A complete pitch/placement sweep found p40 uniquely fast: p33--p39 and
p41--p42 forms clustered near 22 TFLOPS; the two p40 placements reached 45.812
and 46.024 TFLOPS.

The periodic form compressed only A.  Alternating 24- and 16-half row strides
fits two A buffers plus two ordinary p8 B buffers in exactly 32 KiB while
keeping the static bank-use count balanced.  Both parity orientations were
exact, but reached only 44.912--44.947 TFLOPS.  Static bank counts are therefore
not a sufficient predictor of the gfx1151 LDS schedule.

An independent single-buffer padding audit covered A=1--15 with B=p8 and
B=1--15 with A=p8.  P8 is a singular optimum on both operands.  Odd padding
fell to roughly 32.4--38.0 TFLOPS; non-p8 even B padding reached about
41.2--41.4, and non-p8 even A padding about 37.3--38.1.  There is no missed
asymmetric padding win.

Finally, CPU energy preference was changed reversibly inside one exclusive
GPU bracket while the GPU remained fixed at its high 2.9-GHz state.
Performance/power/balance-power/performance measured
47.763/47.881/47.932/47.862 TFLOPS.  The 0.35% span is far smaller than the
remaining 2.8% target gap; CPU package policy is not a promotion path.

### Resident-fragment, cache-policy, and clause closure

Reversing the retained fragment schedule did not remove its serial LDS cost.
The B-resident form keeps all four B fragments live and streams two A
fragments through the four output rows.  It passed the full reference tuple at
135 VGPR, 22 SGPR, 18 KiB LDS, two blocks/16 waves per CU, and zero spills, but
reached only 44.938 TFLOPS.  The extra live B state and A reload schedule cost
more than the three B load/wait gaps they replace.

Hand-edited cache-policy variants were also exact.  In a single bracket with
48.263 and 47.933-TFLOPS controls, A-only DLC/GLC/SLC reached
47.956/47.386/47.246, B-only DLC/GLC/SLC reached
46.770/47.973/47.721, and SLC on both inputs reached 47.307 TFLOPS.  Default
global-load caching remains the retained policy.

Finally, the three-load clause, a split two-plus-one clause, and no clause all
passed full validation.  Against 48.340 and 48.481-TFLOPS controls they reached
48.387, 48.254, and 48.540 TFLOPS respectively.  The no-clause result is only
0.12% above the closing control and is not distinguishable from same-pass
noise.  Enabling the existing two-B-fragment source schedule produced
byte-identical ROCm 7.14 device instructions to the control, so it was removed
from the GPU queue rather than reported as an independent timing result.

### Prefetch, handoff, and vector-epilogue closure

The next experiments deliberately kept failed designs in the ledger because
they identify which apparent profiler costs are actually recoverable.  Every
timed candidate below passed the same full-reference tuple unless explicitly
identified as an invalid intermediate result.

| Candidate | TFLOPS | Same-pass control | Lesson |
|---|---:|---:|---|
| Direct B / shared A | 32.774 | 48.57 / 48.49 | Duplicate B traffic is not hidden by cache reuse |
| Pair-local A handoff | 44.723 | 48.15 / 48.03 | LDS flag polling costs more than the removed barrier |
| Cyclic pair-local A prefetch | 37.298 | 48.44 / 48.41 | More overlap compounds the software-handoff cost |
| Early flat refill in dedicated VGPRs | 45.714 | 48.19 / 48.28 | Longer live ranges and front-loaded VMEM lose to late reuse |
| Early vector/scalar MUBUF refill | 46.009 / 46.100 | 48.27 / 48.31 | Changing the address path does not rescue early-all prefetch |
| No explicit VMEM wait | 48.179 | 48.32 / 48.15 | The extra source wait is not a material runtime stall |
| Correct vector epilogue | 47.335 | 48.26 / 48.34 | Fewer stores do not repay the transpose instructions |

The pair-local handoff retained two blocks/16 waves, but used 142 VGPR and
24,608 bytes of LDS.  It replaced one workgroup barrier with native LDS flag
stores, scalarized polling, and peer-to-peer publication.  Its exactness proves
the protocol, while its 7% regression shows that gfx1151's hardware barrier is
cheaper than this software producer/consumer mechanism.  The early flat form
used 130 VGPR and was 5.2% slower; moving vector or scalar MUBUF refills before
the full 16-WMMA cluster produced similar regressions.  The compiler's late
destination-register reuse is more valuable than maximum nominal load lead.

The first vector-epilogue implementation failed correctness because it treated
DPP `bank_mask` bits as lane-id bits.  On RDNA 3.5 they enable four-lane bank
groups.  The corrected 8x8 transpose uses full-bank `row_xmask` operations plus
lane selection for bits 0 and 1, and 0xa/0x5 bank masks only for bit 2.  It
restored the exact reference tuple, then measured 1.96% below the control
midpoint.  The invalid 47.382-TFLOPS output is not a performance result; the
failure established the ISA rule, and the corrected run closed the epilogue as
a route to 50 TFLOPS.

Scalar-offset MUBUF recurrence is the only repeatable positive signal in this
set.  The short bracket reached 48.692 TFLOPS against 48.516/48.544 controls.
Four longer interleaved runs averaged 47.786 TFLOPS versus 47.613 for their
controls, a smaller but consistent **0.36%** uplift.  It stays at 118 VGPR,
22 SGPR, 18 KiB LDS, two blocks/16 waves, and zero spills.  Removing its clause
fell from 48.500 to 48.340 TFLOPS in one bracket.  Moving its SALU recurrence
after the final WMMA group was neutral at 48.298 versus 48.300, while advancing
only the B refill by four/eight WMMA slots reached 48.203/48.386 versus 48.435.
The retained lesson is narrow: scalar address recurrence helps slightly, but a
compact late refill and its clause are more important than extra latency lead.

These results leave the 48.614-TFLOPS p8 single-buffer kernel as the retained
FP16-output leader.  Its 49.573-TFLOPS short maximum and isolated 50+ samples
remain non-promotable; no sustained 50-TFLOPS result has been recorded.

Both queued cache mappings passed full-reference validation but regressed:
plain 5x8 reached 46.640 TFLOPS and cyclic-skewed 5x8 reached 45.985 TFLOPS,
against 47.793 TFLOPS in the same control pass.  `s_setprio` was worse at
42.439 versus 47.937 TFLOPS.  Moving the LDS wait from before to immediately
after `s_barrier` was neutral at 47.789 versus 47.889 TFLOPS.  Split
`s_barrier_signal`/`s_barrier_wait` forms cannot be assembled for gfx1151;
LLVM reports both instructions unsupported.

One further barrier-removal architecture made each wave's 64x16 A tile private
and ping-ponged only the shared B tile.  It fits exactly 32 KiB LDS and compiles
at 123 VGPR with no spills, preserving two blocks/16 waves per CU.  The price
is duplicate A traffic for the two N waves in each row band.  It passed the
exact full-reference gate but reached only 37.707 TFLOPS, showing that traffic
and private-tile addressing cost substantially more than the removed barrier.

### IU4 instruction qualification

The RDNA3.5 ISA and AMD Matrix Instruction Calculator identify
`v_wmma_i32_16x16x16_iu4` as a distinct signed-or-unsigned INT4 matrix
instruction: two packed operand VGPRs per lane, eight I32 accumulator VGPRs,
8192 integer operations in 16 cycles, and a nominal 2048 operations/WGP/cycle.
The `NEG[0:1]` fields select A/B signedness; wave32 repeats A and B across its
two 16-lane halves.  The result mapping puts even rows in lanes 0-15 and odd
rows in lanes 16-31.

`tools/bench_wmma_iu4.hip` now checks that mapping against a nonuniform exact
16x16 signed product and separately measures independent accumulator chains.
All mapping and all-ones arithmetic checks pass.  The chain sweep measured
103.040, 98.107, 108.380, 109.408, 109.718, and **110.229 INT4 TOPS** for
1, 2, 4, 8, 12, and 16 chains respectively.  The best result is 92.8% of the
118.8-TOPS clock-derived gfx1151 ceiling.

This does not solve the FP16 record by substituting the unit.  The instruction
accepts linear signed/unsigned four-bit integers, while ROCmFP4 Codebook10 uses
the nonlinear levels `0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10`; its
positive 8/10 and negative 10 are outside one signed nibble.  An exact direct
sum-of-linear-IU4 representation therefore needs at least two weight-side IU4
products; a one-WMMA form would still need nonlinear correction work outside
WMMA.  The two-product route leaves a best-case measured equivalent rate near
55 TOPS before activation quantization, scale application, memory traffic, and
output conversion.  The credible IU4 route is a separately quality-gated
linear W4A4 format, not a bitwise reuse of the current ROCmFP4 weights and not
a TFLOPS claim.

The first full W4A4 architecture is now measured rather than inferred.  It
uses a 128x128 workgroup tile, eight waves, eight independent I32 accumulator
fragments per wave, block/K-major packed operands, double-buffered 4 KiB LDS,
and one barrier per K16.  The conflict-free LDS row is exactly two DWORDs: the
16 unique operand lanes cover all 32 banks once, while lanes 16-31 request the
same addresses for multicast.  It compiles at 89 VGPR and 22 SGPR with no
spills.

At 4096 cubed it reached **85.907 INT4 TOPS** in 1.599854 ms and matched every
one of 16,777,216 structured-reference INT32 outputs exactly.  Two bracketed
same-pass runs of the four-wave-column geometry reached 85.063 and 85.907
TOPS, versus 84.992 and 84.182 for the transposed two-column geometry.  A
12-byte padded LDS row was slower at 83.416 TOPS.  Doubling the M tile to
256x128 doubled the per-wave accumulator set, raised allocation to 159 VGPR,
and regressed to 78.367 TOPS.  This makes the 128x128, four-column form the
retained W4A4 architecture.

### Paperclip literature pass and four-wave live-range split

Paperclip full-text extraction was used for a focused pass over recent kernel
work. Five findings map directly onto the gfx1151 frontier:

- The FP16 study in [Hand-Written PTX Tensor-Core GEMM
  Kernels](https://arxiv.org/abs/2608.10103) found that halving accumulator
  register pressure and raising occupancy did not improve FP16 GEMM, while a
  deeper pipeline could lose to memory-queue pressure. This reinforces the
  existing rule that a lower VGPR count is only a hypothesis until same-pass
  timing and counters confirm it.
- [Nautilus](https://arxiv.org/abs/2604.14825) explicitly uses buffer-lifetime
  analysis, live-range splitting, and rematerialization near use to reduce
  local-memory pressure. That transformation inspired the partial late refill
  below.
- [VeriLocc](https://arxiv.org/abs/2506.17506) demonstrates that register
  assignment itself can expose performance missed by a production compiler.
  Its reported MI250x gain relies on CDNA2 AccVGPR placement and does not
  transfer directly to RDNA3.5, but its method motivates treating address and
  fragment allocation as an optimization dimension rather than fixed output.
- [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262) models optimized-kernel
  performance through tile-level data and order dependencies. For this kernel
  the relevant unit is therefore the complete `load -> WMMA -> handoff` graph,
  not the nominal latency of one load or barrier in isolation.
- [TileFuse](https://arxiv.org/abs/2606.11357) reports that offline pre-tiling
  should follow the kernel's physical consumption order and place quantization
  metadata beside the weight tile that consumes it. Its XDNA2 microkernels are
  not portable to gfx1151, but the layout rule is: prepack weights and scales
  together for the inference kernel instead of paying runtime gather or
  materialization costs.

The new four-wave 128x128 experiment applies those lessons without changing
the default kernel. A compiler scheduling fence after each B fragment keeps
only one B fragment live and reduces static allocation from 153 to 129 VGPR.
The next step overlaps only the two A refill vectors with WMMA, commits A after
the handoff barrier, and then loads and immediately commits B. This splits the
A/B refill live ranges and reaches 121 VGPR. Finally, scalar-resource MUBUF A
loads share one vector offset and encode the second 128-bit load with an
immediate `+16`, reaching **120 VGPR**, 22 SGPR, 12 KiB LDS, and zero spills.

The allocation boundaries matter more than the raw counts: the RDNA3.5
24-VGPR wave32 quantum puts 153 in the 168-register class and 129/127 in the
144-register class; 120 reaches the next class. ROCm 7.14 device assembly for
the default path is byte-identical after normalizing the per-build HIP CUID.
The GPU bracket resolved the tradeoff. The 153-VGPR control reached 41.419
TFLOPS at four blocks/16 waves, while the 129-VGPR streamed-B form reached
45.694 TFLOPS at five blocks/20 waves, a real 10.3% architecture gain. The
later forms all retained five blocks/20 waves and passed the exact full
reference tuple, but lost speed as overlap was removed: late-B1 at 127 VGPR
reached 45.011 TFLOPS, split refill at 121 VGPR reached 44.630, and the
120-VGPR MUBUF split reached 44.839. The bracketed 129-VGPR controls reached
45.585/45.598 and the retained p8 controls reached 48.031/48.048 TFLOPS.

The first bracket isolated only one side of a joint allocation question. The
12-KiB tile prevents the 120-VGPR form from admitting a sixth block, while the
129-VGPR streamed form remains in a higher register class. A follow-up padding
sweep therefore reduced LDS and register pressure together.

Lowering LDS alone did not help: streamed-B p8p2, p4p4, p2p0, and p0p0 used
10.5, 10, 8.5, and 8 KiB and reached 40.608, 26.696, 37.495, and 45.445
TFLOPS, all at the host API's reported five blocks/20 waves. The p0p0 result is
important because it retained the p8 streamed control's speed while crossing
the LDS threshold; the other pitches exposed severe bank-phase penalties.

Combining the split MUBUF refill with those layouts compiled at 124/122/120/119
VGPR. The p8p2 and p4p4 forms reported six blocks/24 waves, yet reached only
40.854 and 26.495 TFLOPS--essentially unchanged from their five-block
counterparts. The p2p0 and p0p0 forms unexpectedly still reported five
blocks/20 waves and reached 35.426 and 43.796 TFLOPS. All eight padding
candidates passed the full rocBLAS reference tuple.

This failure resolves two ambiguities. First, additional nominal occupancy
does not recover throughput when the LDS access phase is poor. Second, static
`.vgpr_count` is not a sufficient occupancy oracle on this code object: it
falls to 119 while `.amdhsa_next_free_vgpr` remains 169 for all four combined
forms, and the runtime occupancy result is non-monotonic. Use the runtime query
as a screening observation, not proof of active hardware waves; only timing and
counters can establish the mechanism. The 129-VGPR p8 streamed-B form remains
the best four-wave result at about 45.7 TFLOPS, 5.1% below the retained p8
leader, and no lower-padding form advances to a longer promotion run.

### Hybrid single-operand ping-pong

The next architecture tested whether the full double buffer was solving too
much. The retained kernel has a 12-KiB A tile and a 6-KiB B tile. Buffering
only A therefore uses 30 KiB LDS; buffering only B uses 24 KiB. Both preserve
two 8-wave blocks in the 64-KiB CU budget, and both buffer displacements are
bank-phase neutral at p8.

The A-only form loads all current A fragments, streams one B fragment at a
time, and writes the next A tile to the inactive buffer inside the WMMA
cluster. Only B remains in the serial overwrite handoff. LLVM initially
extended fragment lifetimes to 151 VGPR; a scheduling fence after each B group
restored 127 VGPR, 22 SGPR, and zero spills. Compile-time phase unrolling made
allocation worse at 177 VGPR and was rejected before GPU use.

Three exact A-only schedules were bracketed. Split A loads reached
44.921/44.619 TFLOPS, grouped loads reached 44.606/44.569, and a late
`vmcnt(1)` form--which gives A twelve WMMAs of lead while allowing B to remain
outstanding--reached 44.606/44.747. Same-pass p8 controls were
48.082/48.079 and 48.045/47.809 TFLOPS. All candidates reported two
blocks/16 waves and reproduced the full rocBLAS error tuple.

The B-only sibling issues B before both A loads, uses `vmcnt(2)` to retire only
that oldest operation, and stores B to its inactive buffer while A stays in
flight. It compiled at 129 VGPR, 22 SGPR, 24 KiB LDS, and zero spills. It also
reported two blocks/16 waves and was exact, but reached only 45.178/45.202
TFLOPS against 47.872/48.009 controls.

This closes partial ping-pong for the retained geometry. Moving one or two LDS
writes into the compute cluster does not remove either workgroup barrier; it
adds an extra wait threshold and competes with the 32 fragment reads. The
result is a 5.8--7.3% regression even though occupancy, p8 bank phase, global
traffic, and arithmetic are unchanged. The experiment remains opt-in and the
default device code is instruction-identical after CUID/comment normalization.

### P8 two-address LDS qualification

The closest earlier paired-LDS result used p4 because
`ds_load_2addr_b64` encodes two unsigned eight-bit offsets in eight-byte units.
At p8, the fourth 16-row fragment starts at unit 288 and cannot use the
original base directly. A deterministic assembly transform now creates two
loop-invariant bases shifted by 512 bytes (64 units), placing those final
fragment offsets at 224--227. This adds two address VGPRs outside the loop and
assembles at 121 VGPR, 22 SGPR, 18 KiB LDS, and zero spills.

The all-paired form converts both fragment reads and cooperative refill stores
to `ds_*_2addr_b64`. It passed the full rocBLAS tuple and the host API reported
three blocks/24 waves, up from the retained kernel's two blocks/16 waves. The
extra nominal residency did not help: two runs reached 44.468/44.428 TFLOPS
against 47.914/47.991 same-pass p8 controls.

A read-only isolation restored the three native `ds_store_b128` refill stores
while retaining the paired loads. It remained exact at the same resource and
occupancy report, but reached 43.442/42.041 TFLOPS in a lower-clock bracket
whose controls reached 44.932/45.055. The absolute values from that bracket
must not be mixed with the preceding pass; both same-pass comparisons still
reject the candidate.

The p4 two-address improvement therefore does not transfer to p8. The p8
kernel's native contiguous 128-bit LDS transaction is part of its singular
bank/issue optimum; replacing it with paired 64-bit addresses loses 7.3% in
the clean bracket despite more reported waves. Restoring native stores makes
the result worse, so the read transaction--not merely the handoff store--is
the closed frontier. The shifted-base transform is retained to prevent the
offset-encoding limitation from being mistaken for an untested opportunity.

### Progressive post-barrier refill commit

The retained single-buffer loop originally retires all three refill VMEM
operations before the overwrite barrier, then writes both A vectors and the B
vector to LDS. A source-level attempt to move the wait failed because LLVM
inserted its own `s_waitcnt vmcnt(0)` ahead of the barrier. The reproducible
assembly transform in `tools/patch_progressive_commit_asm.py` removes both
full waits and commits completed operations in program order after the
barrier: `vmcnt(2)` then A0, `vmcnt(1)` then A1, and `vmcnt(0)` then B. The
final LDS wait and publish barrier remain unchanged.

The transformed kernel is exact at 118 VGPR, 22 SGPR, 18 KiB LDS, zero spills,
and the same reported two blocks/16 waves as the leader. A short bracket
reached 48.132/48.411 TFLOPS against 48.158/48.193 controls. Four longer
interleaved runs averaged **47.948 TFLOPS** against their immediately preceding
controls at **47.790 TFLOPS**, a repeatable **+0.158 TFLOPS (+0.33%)** signal.
Every run reproduced the full rocBLAS error tuple.

This is an occupancy-neutral code-generation building block, not a new record.
It shows that the refill queue can make partial forward progress across the
overwrite barrier without changing the memory contract, but its gain is the
same small scale as scalar-offset MUBUF addressing. The two changes are
independent, and their combined assembly confirms that the gains compose. A
short bracket averaged 48.701 TFLOPS for progressive commit plus scalar-offset
MUBUF, versus 48.461 for progressive commit and 48.258 for the controls.

The longer interleaved screen made the signal clearer. Four combined runs
reached 48.317/47.990/48.138/48.247 TFLOPS, averaging **48.173**. Two
progressive-only runs averaged 47.812, and three controls averaged 47.756.
The combination therefore gained **0.75%** over progressive-only and **0.87%**
over the same-pass controls while preserving exactness and resources. This is
the new research base, but not a promoted result: the pass ran below the
historical 48.614-TFLOPS sustained leader and no sample reached 50 TFLOPS.

### One-fragment B lookahead and register placement

The combined base still loads each of the last three B fragments immediately
before its four WMMAs and waits for both 128-bit LDS reads. A hand transform
introduced one alternate eight-VGPR B bank, preloaded B1 with the initial A/B
cluster, then loaded B2 while consuming B1 and B3 while consuming B2. The
instruction count and LDS layout are unchanged; the intended difference is
that each future B load overlaps four current WMMAs.

The first transform failed validation because only the first WMMA after the
interleaved global refill was redirected to B3; the remaining three still read
B2. Its deterministic cosine 0.813 output and 45-TFLOPS timing are invalid,
not performance evidence. Redirecting all four consumers restored the full
rocBLAS tuple. This failure is retained because it demonstrates that a WMMA
fragment remains live across the interleaved refill instructions, not merely
up to the first matrix instruction.

Four exact placements then swept the alternate B base while preserving the
schedule:

| Alternate B base | VGPR | Reported occupancy | TFLOPS |
|---:|---:|---:|---:|
| 118 | 126 | 3 blocks / 24 waves | 45.419 |
| 120 | 128 | 3 blocks / 24 waves | 45.999 |
| 122 | 130 | 2 blocks / 16 waves | 45.203 |
| 124 | 132 | 2 blocks / 16 waves | 45.929 |

Opening and closing combined-base controls reached 48.352/48.276 TFLOPS.
Changing the register phase moves throughput by 1.3% within the 24-wave class
and 1.6% within the 16-wave class, confirming register assignment as a real
gfx1151 tuning dimension. It cannot rescue this dataflow: the best placement
is still 4.8% below the control. The deeper initial LDS queue and alternate
WMMA operand bank cost more than the three hidden wait gaps save, and additional
reported residency again fails to predict throughput.

### Semantics-preserving VGPR boundary phase

The B lookahead sweep proved that physical register assignment changes WMMA
throughput, but its dataflow confounded placement with an alternate live B
bank. A cleaner transform keeps every instruction and dependency unchanged:
accumulators remain in v1--v64, while every address, A/B fragment, and refill
register at v65 or above is shifted together by a small gap. This changes the
operand-to-accumulator bank phase without changing relative live ranges.

Odd shifts were rejected statically because an existing dual-VALU pair then
places both destinations on even registers; gfx1151 requires one even and one
odd destination. Splitting that instruction would confound the sweep, so only
the instruction-identical deltas 2/4/6 were assembled. All use 22 SGPR,
18 KiB LDS, and zero spills.

The exact short bracket was sharply phase-selective:

| Boundary delta | VGPR | Reported occupancy | TFLOPS |
|---:|---:|---:|---:|
| 0 control | 118 | 2 blocks / 16 waves | 48.399 / 48.381 |
| 2 | 120 | 2 blocks / 16 waves | **49.439** |
| 4 | 122 | 3 blocks / 24 waves | 45.049 |
| 6 | 124 | 3 blocks / 24 waves | 46.025 |

Delta 2 then averaged 49.323 TFLOPS across four longer interleaved runs versus
48.192 for their immediately preceding controls (+2.35%). The strict record
screen used five fresh processes, 20 warmups, and five timing blocks of 100
iterations each. Their medians were **49.095, 48.980, 48.995, 48.986, and
49.116 TFLOPS**, for a **49.035-TFLOPS average** and **48.980-TFLOPS floor**.
Opening/closing combined-base controls reached 48.103/47.941. Every process
reproduced the full rocBLAS tuple.

This promotes delta 2 as the new sustained prepacked-contract leader: +2.11%
over its same-pass control average and +0.87% over the former 48.614-TFLOPS
leader. It is not a 50-TFLOPS result; the remaining sustained gap is 1.97%.
The delta-4/6 collapse despite 24 reported waves also makes the mechanism
clearer: physical WMMA register phase dominates the nominal occupancy change.

A follow-up held delta 2 constant and moved the insertion boundary across every
safe live-range cut at v17/v33/v49/v57/v66/v68/v69/v70. The host entered a
lower package state in this pass: opening/closing v65 controls reached only
45.510/45.302 TFLOPS, so these absolute values must not be compared with the
49.035 qualification. Same-pass results were
44.430/44.461/44.464/44.848/45.150/44.420/44.513/44.607 TFLOPS in boundary
order. All were exact. V66 came closest but remained 0.56% below the control
midpoint; every other cut lost 1.2--2.2%. The winning transformation therefore
includes v65 and all later address/fragment registers as one phase group.

Two finer placement sweeps then held the delta-2 dataflow and 120-VGPR count
fixed. The hot-loop sweep exchanged B's eight-register bank with each of the
four A banks. The epilogue-safe accumulator sweep exchanged 16-register pairs
at v17, v33, and v49; pairs involving v1 were rejected because the epilogue
requires contiguous `v[0:1]`. Every assembled permutation was an exact
involution and reproduced the full rocBLAS tuple.

The first bracket suggested small positives: B/A0 reached 49.767 TFLOPS and
the v17/v49 accumulator exchange reached 49.785, against 49.580/49.622
controls. A composition bracket reversed those signs. Its controls reached
49.863/49.827 TFLOPS, while B/A0 reached 49.598, v17/v49 reached 49.567,
B/A0 plus v17/v49 reached 49.552/49.886, and B/A3 plus v17/v49 reached 49.758.
The best combined average remained below the control midpoint. These are
order/noise effects rather than additive gains; the unpermuted delta-2 form
remains the leader.

## Counter-guided follow-up — 2026-08-24

The retained delta-2 leader was profiled with ROCm 7.14 rocprofiler-compute
using a separate counter pass; counter-pass durations are not throughput
denominators. The exact leader dispatch remained 120 VGPR, 22 SGPR, and 18 KiB
LDS with two resident blocks. Its two counter passes reported 47.747 and
47.588 TFLOPS under profiling perturbation; both exactness tuples were unchanged
(`normalized_max_error=0.018779343`, `cosine_similarity=0.999977929`). The
first pass reported `SQ_WAIT_BARRIER=1.212e9`, `SQ_WAIT_CNT_ANY=6.537e8`, and
`SQ_WAIT_INST_LDS=3.879e8`, while `GRBM_GUI_ACTIVE` remained close to the
dispatch interval. This points to synchronization and operand retirement as
the useful frontier, not a missing DRAM transfer.

The following architecture-level tests were then run or rejected at compile
time:

| Candidate | Result | Decision |
|---|---:|---|
| K32 two-slot persistent ring | 39.561 TFLOPS; exact | Reject: the larger K32 schedule loses to K16 |
| Source `WAIT_AFTER_BARRIER` | 47.140 TFLOPS; exact | Reject: no improvement over source control |
| Source `NO_EXPLICIT_VMWAIT` | 47.094 TFLOPS; exact | Reject: no improvement over source control |
| Remove publish barrier | 47.360 TFLOPS; normalized error 0.562 | Reject: synchronization is required |
| Stream one A fragment at a time | 135 VGPR, no spills | Reject before timing: register pressure increased |
| M-major WMMA issue order | 142 VGPR, no spills | Reject before timing: register pressure increased |
| Persistent tile scheduler | 35.821 TFLOPS; exact; one block/eight waves | Reject: the long square GEMM is already compute-saturated |
| Hot-loop compare elision | GPU page fault before validation | Reject: SCC cannot be carried safely across the refill/WMMA sequence |

The barrier-removal failure is particularly useful: `s_waitcnt` retires a
wave's own LDS writes but does not publish them to the other waves, so it
cannot replace the second workgroup barrier. The source-level register tests
also confirm that apparent A-fragment hoisting is not the sole cause of the
120-VGPR allocation; restructuring the loops extends other fragment and
prefetch live ranges. The delta-2 assembly leader therefore remains the control
for the next hand-scheduled register/WMMA experiment.

A refill-order screen then moved the next B global load and its post-barrier
LDS stores ahead of the two A fragments (`B,A0,A1` instead of `A0,A1,B`).
The transformation preserved the 120-VGPR/22-SGPR/18-KiB-LDS resource tuple
and the complete reference error tuple. Five fresh processes measured
48.846, 48.889, 48.798, 48.690, and 48.784 TFLOPS (48.801 average,
48.690 floor), below the delta-2 control and therefore rejected. The apparent
positive single-pass signal was scheduling/power noise, not a transferable
gain; the original refill order remains the comparison baseline.

The previously compiled N-packed/N-major family was finally screened on the
GPU rather than left as a static-code hypothesis. The base, M-major,
2x4-warp/128x256, and K32 forms measured 42.522, 39.857, 40.919, and 38.695
TFLOPS respectively under the original-layout contract; all were exact, but
none transferred to the retained prepacked path. A hand-injected `s_setprio 1`
around the delta-2 WMMA cluster (with `s_setprio 0` before the handoff barrier)
also preserved 120 VGPR and exactness but fell to 42.595 TFLOPS. Finally,
interleaving the refill as `A0,B,A1` reached 48.816 TFLOPS, exact but below
the delta-2 leader. These close the remaining compiled schedule variants.

A new LDS issue-order experiment then issued the first B fragment before the
A fragments, because that B operand feeds the first WMMA cluster. It preserved
120 VGPR, 22 SGPR, 18 KiB LDS, and exact output. In a five-process paired
screen it measured 49.276, 49.094, 49.010, 49.019, and 48.982 TFLOPS (49.076
average, 48.982 floor), versus fresh delta-2 controls at 49.223, 49.211,
49.201, 49.124, and 49.052 (49.162 average, 49.052 floor). The reordered
critical path therefore loses 0.086 TFLOPS and is rejected; the apparent
single-screen 48.985 result was package-state noise.

Relaxing the hot-loop LDS waits by one (`lgkmcnt 4->5` and `2->3`) looked
promising at 49.117 TFLOPS, but failed the numerical gate catastrophically:
normalized maximum error 137.345592071 and cosine 0.693088403. The existing
wait thresholds are therefore load-bearing and remain unchanged.

A synchronization reduction removed only the `lgkmcnt(0)` wait immediately
before the publish barrier, retaining the VMEM wait and both workgroup
barriers. It remained exact with unchanged resources but reached 48.904
TFLOPS. The wait is therefore not redundant on gfx1151; the original
pre-barrier retirement sequence remains the control.

Moving the intermediate `lgkmcnt(2)` wait one WMMA later reached 49.073
TFLOPS, but failed validation (`finite=no`, normalized maximum error
0.100156495, NaN RMS/cosine). The current wait placement is a real fragment
dependency boundary and is retained.

Reversing each contiguous independent WMMA run in the hand-assembled hot loop
preserved all waits, loads, barriers, resources, and exact output, but reached
48.809 TFLOPS. Matrix issue order is therefore not a free throughput gain; the
original compiler-derived order remains the control.

Additional compiled B-phase and loader variants were screened to close the
remaining binary backlog. `bp-b-resident` reached 44.600 TFLOPS exact;
`bp-load-bmiddle` reached 48.672 exact; and `bp-load-astripe` was numerically
invalid (normalized maximum error 208.326). The `bp-bphase-asm16` and
`bp-bphase-asm32` objects failed occupancy with invalid device functions
before timing. None is a candidate for the prepacked leader.

The prefetch-permutation backlog was also screened. `bp-pf-013`, `021`,
`023`, `032`, and `123` measured 47.649, 47.587, 47.531, 47.473, and
47.308 TFLOPS respectively, all exact. The lower-occupancy `bp-prefetch-012`,
`021`, `102`, and `120` forms measured 37.032, 37.227, 37.313, and 38.031
TFLOPS exact with three resident blocks. These permutations are closed and do
not advance the delta-2 control.

VMEM clause/grouping variants were then screened: `bp-clause2-break1`,
`bp-clause3`, `bp-no-clause`, `bp-no-explicit-vmwait`, and `bp-no-vmwait`
measured 47.723, 47.662, 47.589, 47.501, and 47.445 TFLOPS respectively.
All were exact at the same 120-VGPR occupancy, but every form regressed from
the delta-2 schedule.

## Decision

The correct block/K-major p8 kernel with progressive refill, scalar-offset
MUBUF, and the delta-2 VGPR boundary shift now averages 49.035 TFLOPS across
five fresh 100-iteration processes, with a 48.980-TFLOPS floor. It remains a
distinct persistent-input contract and has not met the 50 TFLOPS promotion
gate. The research kernels and negative results are kept for reproducibility.
Continue from the measured delta-2 register phase; do not promote isolated 50+
samples that fail the sustained same-pass control.
