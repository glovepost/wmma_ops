# RDNA3.5 / WMMA reference material — annotated

Sources ranked by how much weight they can carry, with what each is actually
authoritative *for*. Several widely-cited sources are wrong or misleading on
specific points; those are flagged rather than omitted, because they keep being
found.

Context: gfx1151 (Strix Halo, RDNA3.5), wave32, `v_wmma_f32_16x16x16_f16`.

---

## Tier 1 — primary, verified

**AMD "RDNA3.5" Instruction Set Architecture Reference Guide** (doc 70649,
2024-07-23). The ground truth.
- §7.9 WMMA: operand table, the lane-replication rule verbatim, round-to-nearest-even.
- §7.9 **figures**: "A / B / C & D Matrix: VGPR View for Wave32" — the full
  element-to-register mapping. **Vector figures: invisible to `pdftotext` and to
  any grep.** Transcribed in our markdown conversion.
- §7.9.1 WMMA Scheduling: back-to-back dependent WMMA needs one `V_NOP` or an
  independent VALU op between them when the first instruction's D overlaps the
  second's A or B.
- §12.1 LDS: *"64 banks of DWORD-wide RAMs ... sub-divided into two sets of
  32-banks each"*, *"DWORDs are placed in the banks serially"* → `bank = (addr/4) % 32`.

**AMD Matrix Instruction Calculator** — <https://github.com/ROCm/amd_matrix_instruction_calculator>
Authoritative for element↔register mappings and instruction characteristics.
Pass the **gfx target**, not the generic arch: `-a gfx1151`. Its table maps
`gfx1150/1151/1152/1153` onto the `rdna3` matrix model and output is
byte-identical, so results are AMD's answer for the part.
```
matrix_calculator.py -a gfx1151 -i v_wmma_f32_16x16x16_f16 -d
  Execution cycles: 32     FLOPs: 8192     FLOPs/WGP/cycle: 1024
  Can co-execute with VALU: False
matrix_calculator.py -a gfx1151 -i v_wmma_f32_16x16x16_f16 -A -M -w 32
  lane 29 -> A[13][*], lane 30 -> A[14][*], lane 31 -> A[15][*]
```
Needs `tabulate`.

**AMD composable_kernel** — canonical location is now
<https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel>.
`ROCm/composable_kernel` is **deprecated** ("Moved to ROCm/rocm-libraries repo",
develop kept read-only), so prefer the new path; raw fetches from the old one
still resolve and will silently go stale.

AMD's production WMMA implementation, and the single most useful source here.

- `include/ck/utility/amd_wmma.hpp` — the intrinsic wrappers. Its `__gfx11__`
  macro **explicitly lists `__gfx1151__`** alongside gfx1100-1103 and
  gfx1150/1152/1153, so gfx1151 is a first-class WMMA target in AMD's own
  library, not an inherited one.
- `include/ck/tensor_operation/gpu/warp/wmma_gemm.hpp` — carries an ASCII layout
  diagram in-source: for WAVE32, registers `RC0..RC7` down the rows and threads
  across the columns, SubGroup 0 = lanes 0-15, SubGroup 1 = lanes 16-31.
  Independent third confirmation of the C/D layout.

**Attention on gfx11 — the closest public prior art to a decode kernel:**

```
include/ck/tensor_operation/gpu/device/impl/
    device_multi_query_attention_forward_wmma.hpp     <- MQA: one K/V head
    device_grouped_query_attention_forward_wmma.hpp   <- GQA
    device_batched_gemm_softmax_gemm_permute_wmma_cshuffle.hpp
include/ck/tensor_operation/gpu/grid/
    gridwise_batched_gemm_softmax_gemm_wmma_cshuffle.hpp
example/32_batched_gemm_scale_softmax_gemm/
    multi_query_attention_forward_wmma_fp16.cpp       <- instantiated tile configs
    grouped_query_attention_forward_wmma_fp16.cpp
    self_attention_forward_wmma_fp16.cpp
    cross_attention_forward_wmma_fp16.cpp
```

The MQA header states our exact shape:

```text
// Multi-Query Attention (MQA) kernel implementation
// Assume number of head of K,V is 1.
// Q [G0, G1, M, K] * K [G0, 1, K, N] = P [G0, G1, M, N]
// P [G0, G1, M, N] * V [G0, 1, N, O] = Out [G0, G1, M, O]
```

MLA decode is exactly this: one latent K/V head shared across every query head.
The tile shapes AMD instantiates for gfx11 are worth reading before choosing our
own — the smallest is `BlockSize 32` (a **single wave32**) with Gemm0
`MPerBlock=16, LPerBlock=128, KPerBlock=64`, Gemm1 `NPerBlock=64, LTile=64`,
WMMA 16x16x16, and repeats `MRepeat=1, LRepeat=8, NRepeat=4` — i.e. 16 query
rows and 128 KV rows per block, one WMMA M-tile, iterating over L. That is a
very different decomposition from a 256-thread block holding G heads, and it is
AMD's answer for this instruction on this architecture.

Forward only; backward is unsupported on gfx11.

---

## Tier 2 — official, but partial or misleading

**GPUOpen, "How to accelerate AI applications on RDNA 3 using WMMA"** —
<https://gpuopen.com/learn/wmma_on_rdna3/>
The code is correct; **the prose is not**. It describes matrix A as "each lane
stores one column", which reads as the opposite of the truth. Per-lane, A is held
by **row**. The confusion is that the ISA's "row/column major" labels describe
what a *VGPR* holds, not what a *lane* holds. Both statements describe the same
table. This has cost this repo two bug-fix commits.

**rocWMMA** — <https://github.com/ROCm/rocWMMA> (now folded into `ROCm/rocm-libraries`).
Supports gfx1151; minimum ROCm 6.4.

**ROCm "AMD RDNA3.5 system optimization"** —
<https://rocm.docs.amd.com/en/latest/how-to/system-optimization/rdna3-5.html>
Despite the title, **contains no architecture**: it is GTT/TTM/VRAM memory
configuration and minimum kernel versions. Do not go here for CU counts, LDS
geometry or occupancy guidance.

**ROCm blog, "Attention Decode on AMD MI450 — Gluon Kernel Optimization Guide"** —
<https://rocm.blogs.amd.com/software-tools-optimization/gluon-attention-decode-mi450/README.html>
The best structural guide to decode attention: split-K partitioning, two WMMA ops
(QK and PV), transposing WMMA output layouts to avoid conversion between them,
triple-buffered LDS, 85% of peak bandwidth achieved. **But it is CDNA.** Its
central scheduling advice — split softmax into two stages so WMMA and VALU
interleave — does not transfer: on RDNA3 the calculator reports
`Can co-execute with VALU: False`. Its triple-buffering advice also inverts on
gfx1151, where LDS capacity gates occupancy (measured -18% for double buffering
on a bandwidth-bound decode kernel).

---

## Tier 3 — known gaps and community work

**ROCm issue #6025, "Documentation Gap: WMMA Output Lane Mapping for gfx12"** —
<https://github.com/ROCm/ROCm/issues/6025>
Confirms the trap is real and known: the output lane mapping is documented
neither in the ISA text nor GPUOpen, developers reverse-engineer it from CK
source or by identity-matrix probing, and the reporter hit silent transposition.
No AMD response. Gives the **gfx12** mapping:
`VGPR[lane][j] = matrix[(lane/16)*8 + j][lane % 16]`.

> **Porting hazard.** RDNA3 and RDNA4 differ here. RDNA3: lanes hold 16 elements
> and lanes 0-15 are **replicated** into 16-31; C/D rows interleave as
> `C[2j + lane/16][lane%16]`. RDNA4: 8 elements per lane, no replication, rows
> group as `(lane/16)*8 + j`. Both are column-distributed, but code that assumes
> one row grouping silently transposes on the other.

**JohnTDI-cpu/rdna4-wmma-guide** — RDNA4 lane mapping plus a fused MXFP4 GEMM at
40.8 TFLOPS. Adjacent architecture; useful as the RDNA4 half of the contrast above.

**Repeerc/flash-attention-v2-RDNA3-minimal** — FA2 forward with rocWMMA on RDNA3,
aimed at Stable Diffusion. +10% at 512², +40% at 1024², 28-36% VRAM reduction. No
backward pass; causal masking unoptimised; BF16 known slow. Closest public prior
art for WMMA attention on this architecture.

**llama.cpp** — `ggml/src/ggml-cuda/fattn-wmma-f16.cuh`, enabled with
`-DGGML_HIP_ROCWMMA_FATTN=ON`. Issue #26220 reports the native MMA FA kernel
regressing up to 2× against the removed rocWMMA path at depth on RDNA4, while
decode was +3.9% — i.e. the regression is prefill-specific. Worth reading before
assuming native WMMA beats rocWMMA.

**Chips and Cheese, "Microbenchmarking AMD's RDNA 3"** —
<https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture>
Independent measurement. Confirms LDS at 128 KB/WGP as two 64 KB blocks of 32
banks each — matching ISA §12.1. Scalar L0 15.4 ns, vector L0 32 KB/CU 32-way,
L1 256 KB/array, L2 6 MB. FP32 5-cycle latency, dual-issue VOPD compiler-limited.
**No WMMA measurements** — the gap in public microbenchmarking is real; the
"Dissecting Tensor Cores via Microbenchmarks" methodology (arXiv 2206.02874) is
the closest template if we ever want to fill it.

**Strix Halo specific**
- pytorch issue #171687 — gfx1151 LLM decode reported as ~90% `hipMemcpyWithStream`,
  kernels not compute-bound. Relevant framing for any decode work here.
- kyuz0 Strix Halo toolboxes (llama.cpp and vLLM backend benchmark grids); the
  `-rocwmma-improved` images carry a patch retuning rocWMMA for long-context.
- llm-tracker.info Strix Halo page — collected performance data.
- 40 CU, LPDDR5X-8000, ~256 GB/s theoretical. Our own measurement of *sustained*
  bandwidth is ~148 GB/s.

---

## Algorithmic references (architecture-independent)

- **FlashInfer** — <https://flashinfer.ai/2024/02/02/introduce-flashinfer.html>
  Operational intensity of MQA/GQA decode is `H_qo/H_kv`; route decode through the
  tensor-core prefill kernel. For DS4 MLA that ratio is 64/1.
- **FlashMLA** — <https://github.com/deepseek-ai/FlashMLA> — `BLOCK_SIZE_M = 64`
  (the query-head count), seq-parallel tile scheduler. Hopper, but the shape
  argument is portable.
- **AMLA** — <https://arxiv.org/pdf/2509.25224> — cheaper online-softmax rescaling
  for MLA decode (multiply replaced by add), 1.2-1.8x.

---

## What is still missing publicly

1. No WMMA latency/throughput microbenchmark for RDNA3 beyond the calculator's
   32-cycle figure.
2. No official statement of the C/D output lane mapping in prose anywhere — three
   independent derivations agree, but AMD has never written it down outside a
   figure and a source comment.
3. Nothing on WMMA under LDS-bank pressure, which is where our decode kernel lives.
