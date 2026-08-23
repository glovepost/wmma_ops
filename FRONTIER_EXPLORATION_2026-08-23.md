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

## Decision

The correct block/K-major p8 kernel materially exceeds the original-layout
control and has reached 48.614 TFLOPS sustained in the longer screen and
49.573 TFLOPS in the best short pass.  It remains a distinct persistent-input
contract and has not met the 50 TFLOPS promotion gate, so no published source
record has been changed.  The research kernels and negative results are kept
for reproducibility.  Continue with occupancy-preserving barrier reduction and
codegen scheduling; do not promote isolated 50+ samples that fail the
sustained same-pass control.
