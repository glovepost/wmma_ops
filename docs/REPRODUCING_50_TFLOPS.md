# Reproducing the 50 TFLOPS qualification

This document records the complete procedure used to qualify the
**50.073879 TFLOPS average** result on 2026-08-25. The kernel change was
introduced by commit `c1f49567407e95758700656c94021a33eeda4ddb`.

This is a persistent-input research contract, not an ordinary GEMM API:

- target: AMD Strix Halo, `gfx1151`, CU mode;
- shape: M=N=K=4096;
- input: FP16 A and B, already stored in the benchmark's block/K16-packed
  layout;
- arithmetic/output: FP16 WMMA accumulation and FP16 output;
- timing: one kernel dispatch; input packing is outside the timed region;
- validation: all 16,777,216 FP16 outputs against a rocBLAS reference.

Do not compare this number directly with the repository's ordinary-layout,
FP32-output result or with an end-to-end call that includes packing.

## Qualified environment

The run used the shared gfx1151 host under the repository's exclusive-GPU
protocol. The relevant environment was:

| Component | Qualified value |
|---|---|
| Host kernel | `Linux otheru 7.1.3-201.fc44.x86_64` |
| Container tag | `wmma-record:rocm714-torch213` |
| Container image ID | `sha256:c15b5ed415889ec6457e15d64a935a817acf82cc3ea146246c6b7b1e30533682` |
| HIP | `7.13.99004-3309c6114a` |
| AMD clang | `23.0.0git`, LLVM commit `46fcb339fb61119b337f973c7ca9e710a319fdd0`, patched commit `440716f8b87be9d8e20ed910e10e5b6d14d57cf6` |

The local image had the matching repo digest
`wmma-record@sha256:c15b5ed415889ec6457e15d64a935a817acf82cc3ea146246c6b7b1e30533682`.
The digest is the durable identity; the tag alone is not.

Before building or qualifying, confirm that the tag used internally by
`exclusive-frontiers.sh` still resolves to that image:

```bash
docker image inspect wmma-record:rocm714-torch213 --format '{{.Id}}'
```

## Build the selected image

On the GPU host, check out commit `c1f4956` or a descendant containing the
same kernel and build script. Mount the checkout at `/work`, because the build
script deliberately uses that fixed path:

```bash
docker run --rm --name codex-hot-b-phase-build \
  -v "$PWD":/work -w /work \
  wmma-record@sha256:c15b5ed415889ec6457e15d64a935a817acf82cc3ea146246c6b7b1e30533682 \
  ./build-hot-b-phase.sh
```

`build-hot-b-phase.sh` starts from `traces/bp-register-phase-d2.s`, applies
the LDS-only publication wait, generates B placements at `v108`, `v111`, and
`v112`, assembles gfx1151 code objects, bundles each into the host harness, and
links rocBLAS for the reference calculation. The qualified binary is
`bp-hot-b-phase-111`; `bp-publish-lgkm0` is its selected-image control.

Confirm the selected code-object metadata printed by the build (or inspect
`traces/bp-hot-b-phase-111.s`): 18,432 bytes group-segment LDS, zero private
segment bytes, 22 SGPRs, and 120 VGPRs. The resulting residency reported by
the harness must be two blocks and 16 waves per CU.

## What changed in the kernel

`tools/place_hot_b_phase_asm.py` leaves the four hot A fragments on their
proven phase-4 banks but places the repeatedly loaded B fragment at
`v111:v118` (phase 7). A2 and A3 move into the old B banks, the cooperative B
refill moves to `v72:v75`, and the loop-invariant vector offset and LDS address
are shadowed in `v65` and `v66` before the loop.

The transform does not alter the repeated hot-loop opcode inventory or order:
each two-slice body still contains 16 WMMA instructions, 16 128-bit LDS loads,
three 128-bit buffer loads, three 128-bit LDS stores, and two barriers. It also
does not change memory addresses, wait thresholds, WMMA count, or declared
resources. The measured uplift therefore isolates physical operand-register
placement rather than a hidden occupancy or work reduction.

## Run the exact qualification

The shared host permits only one GPU/model workload at a time. The command
below took `/root/gpu.lock`, stopped production through the fixed-purpose
wrapper used by `exclusive-frontiers.sh`, ran every sample in a fresh Docker
process, and restored production from the script's exit trap:

```bash
ssh -o ControlPath=none gpu "flock -w 7200 /root/gpu.lock bash -lc 'cd /root/wmma-half50-codex && WARMUP=20 ITERATIONS=100 ./exclusive-frontiers.sh bp-hot-b-phase-111 bp-publish-lgkm0 bp-publish-lgkm0 bp-hot-b-phase-111 bp-hot-b-phase-111 bp-publish-lgkm0 bp-publish-lgkm0 bp-hot-b-phase-111 bp-hot-b-phase-111 bp-publish-lgkm0' | tee /root/wmma-results/hot-b-phase111-qualification-20260825.txt"
```

The order alternates which member runs first in each candidate/control pair:

| Pair | First | Second |
|---|---|---|
| 1 | candidate | control |
| 2 | control | candidate |
| 3 | candidate | control |
| 4 | control | candidate |
| 5 | candidate | control |

Each process performs 20 warmups followed by five timing blocks of 100
iterations. The harness reports the median of those five block timings. A
process exit status of 1 means it did not clear the combined correctness and
50-TFLOPS promotion check. `exclusive-frontiers.sh` intentionally accepts that
status so below-threshold controls do not stop the bracket. Therefore, inspect
every `validation` and `promotion` line in the retained output; the wrapper's
own successful exit is not the qualification gate.

## Acceptance criteria and recorded result

Promotion required all five fresh candidate processes to:

1. exceed 50 TFLOPS at the fixed 4096-cubed contract;
2. report the expected two-block/16-wave residency;
3. validate every output as finite and reproduce the accepted numerical tuple.

The candidate medians were 50.014850, 50.135584, 50.146413, 50.025904, and
50.046643 TFLOPS. Their arithmetic mean was **50.073879 TFLOPS**, their floor
was **50.014850 TFLOPS**, and the mean of their median times was 2.744730 ms.
The five paired controls averaged 49.018881 TFLOPS, making the same-pass uplift
2.152%.

All ten processes checked 16,777,216 values and reported:

```text
finite=yes
max_abs_error=0.562500000
reference_max_abs=29.953125000
normalized_max_error=0.018779343
rms_error=0.035428338
cosine_similarity=0.999977929
```

The unedited console output is checked in at
[`results/hot-b-phase111-qualification-20260825.txt`](../results/hot-b-phase111-qualification-20260825.txt).
Verify it with:

```bash
sha256sum results/hot-b-phase111-qualification-20260825.txt
```

Expected SHA-256:
`16cd5054723ec6db71b61ce27b0fe9d7d23f009471b9297d31e815090927c6b7`.

## Interpretation

This qualification establishes a repeatable sustained result for the stated
prepacked FP16-output contract. It does not establish 50 TFLOPS for arbitrary
matrix layouts, include preprocessing cost, or supersede the separately
reported ordinary-layout FP32-output contract. Those boundaries are part of
the result, not caveats to omit when quoting it.
