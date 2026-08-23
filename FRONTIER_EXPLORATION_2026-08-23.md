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
| 256x128 block-packed N, p8 | 48.614 | 2.827130 | Retained leader |
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
128 VGPR with zero spills and 30,720 bytes of LDS, and is queued for its first
exclusive correctness/performance screen.

## Decision

The correct block/K-major p8 kernel materially exceeds the original-layout
control and has reached 48.614 TFLOPS sustained in the longer screen and
49.573 TFLOPS in the best short pass.  It remains a distinct persistent-input
contract and has not met the 50 TFLOPS promotion gate, so no published source
or documentation has been changed.  Continue with occupancy-preserving
barrier reduction and codegen scheduling; do not promote isolated 50+ samples
that fail the sustained same-pass control.
