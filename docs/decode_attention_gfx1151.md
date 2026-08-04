# Decode-Attention Findings on gfx1151

Companion to the GEMM results in the main README. Everything here was measured on
the same Strix Halo part, but in a **different regime**: a memory-bound q=1
attention decode kernel rather than a compute-bound large GEMM. Several of the
repo's conclusions invert in this regime, which is the main reason this file
exists.

**Workload**: DeepSeek-V4-Flash MLA decode. `head_dim = 512`, 64 query heads
sharing **one** latent KV head, `n_kv` from 128 to ~33k, K stored FP16. One query
token per step, so the whole span is streamed for a single row of output.

Full source of the benchmark kernel: `tools/bench_decode_attn.hip` in the ember
repo (standalone, ships its own double-precision CPU reference).

---

## Headline: bank conflicts, a third option

The README weighs LDS **padding** (stride 24) against **XOR swizzle** and picks
padding, 20-21 TFLOPS vs 17-18. Both of those fix conflicts by changing where
data *sits*. For this kernel the conflict came from where lanes *read*, and the
fix was neither: reassign the lane -> dim mapping.

The score phase has `TPR` lanes cooperate on one 512-element dot product. The
obvious partition gives each lane a contiguous chunk:

```cpp
// 8-way bank conflict: consecutive lanes land 64 B apart
for (int t = 0; t < DPT; ++t) {
    const int d = lane * DPT + t;
    partial += qh[d] * __half2float(kr[d]);
}
```

With `TPR = 16` and `DPT = 32`, lanes are 64 B apart, so bank
`(lane*16 + t/2) % 32` collides for every other lane. Striding by `TPR` instead
puts consecutive lanes on consecutive 4-byte words, which is conflict-free and
lets the loads vectorise:

```cpp
// conflict-free, and half2 loads fall out for free
const float2  * q2 = (const float2 *) (q + (size_t) h * D);
const __half2 * k2 = (const __half2 *) (ktile + (size_t) r_local * D);
for (int t = 0; t < DPT / 2; ++t) {
    const int idx = t * TPR + lane;          // interleaved, not blocked
    const float2 qq = q2[idx];
    const float2 kk = __half22float2(k2[idx]);
    partial += qq.x * kk.x + qq.y * kk.y;
}
```

Same dims, same F32 accumulation, deterministic. **+22%** (675.7 -> 555.9 us at
n_kv=8896), and it moved the kernel from 108 GB/s to **131 GB/s**.

Worth noting because the first attempt at vectorising this kept the blocked
mapping (`idx = lane*(DPT/2) + t`) and was *slower* than scalar — the conflict,
not the scalar loads, was the cost. Padding the row stride would not have helped
here: the collision is within a row, not across rows.

---

## Measured: decode kernel vs the path it replaces

`expl_full` is the production explicit attention path it would replace — two F32
GEMMs plus the F16->F32 cast, ring/compressed concat, transposed copy, softmax,
scale and sink concat — taken from rocprofv3 traces of the live server.

| n_kv | decode kernel | expl_full | speedup | GB/s | max abs err |
|------|--------------:|----------:|--------:|-----:|------------:|
| 128  |  21.2 us |  40.1 us | 1.89x |  49 | 4.5e-08 |
| 256  |  30.4 us |  48.3 us | 1.59x |  69 | 7.5e-08 |
| 416  |  41.3 us |  66.7 us | 1.61x |  82 | 6.0e-08 |
| 512  |  46.1 us |  75.3 us | 1.63x |  91 | 4.8e-08 |
| 768  |  61.7 us | 101.6 us | 1.65x | 102 | 4.8e-08 |
| 960  |  80.2 us | 114.9 us | 1.43x |  98 | 4.1e-08 |
| 1616 | 119.3 us |        - |     - | 111 | - |
| 3278 | 218.4 us |        - |     - | 123 | - |
| 8896 | 556.2 us | 1178.4 us | **2.12x** | **131** | - |

Sustained achievable bandwidth on this part measures **~148 GB/s** (independently
from production GEMM traces: 72.9 MB in 492.6 us), against ~256 GB/s theoretical.
So 131 GB/s is **88% of achievable** — comparable to the 85%-of-peak that AMD's
own MI450 decode guide reports.

---

## Where this regime inverts the README

| Optimization | README (GEMM) | Here (decode) |
|--------------|---------------|---------------|
| Double buffering / ping-pong LDS | **+20%** | **-18%** (23.0 -> 27.2 us at n_kv=128) |
| High occupancy vs latency hiding | latency hiding wins | **occupancy wins** |
| XOR swizzle vs padding | padding, +15-20% | neither; fix the lane mapping |

Not a contradiction — the two workloads sit on opposite sides of the same axis.
The GEMM runs at 36% of FP16 peak and is compute-bound, so spending LDS to hide
memory latency pays. The decode kernel runs at 88% of achievable bandwidth, where
LDS capacity gates occupancy directly: doubling LDS for a prefetch buffer halves
resident blocks and costs more than the latency it hides.

The README's "triple buffering: broken" and "BLOCK_K=32: slower" both still hold
in this regime.

---

## Why the head group can't grow yet (and why WMMA is the unlock)

MLA decode has operational intensity `H_qo / H_kv` = 64/1, which is squarely
matrix-core territory (FlashInfer's argument for routing GQA decode through the
tensor-core prefill kernel; FlashMLA independently picks `BLOCK_SIZE_M = 64`,
exactly the query-head count).

`G` = query heads per block sets how many times the KV span is re-read
(`n_heads / G`). Larger `G` should therefore be strictly better. Measured at
n_kv=8896, it is not:

| G | KV traffic | time | effective |
|---|-----------:|-----:|----------:|
|  8 | 72.9 MB | 555.9 us | 131 GB/s |
| 16 | 36.4 MB | 713.8 us |  51 GB/s |
| 32 | 18.2 MB | 786.7 us |  23 GB/s |
| 64 |  9.1 MB | 829.9 us |  11 GB/s |

Traffic falls **8x** and the kernel gets **slower**. Above G=8 nothing is
bandwidth-bound — it is scalar-FMA compute plus register pressure (the
accumulator is `2G` VGPRs/thread; G=64 is 128 VGPRs). The 9.1 MB floor would be
~62 us at 148 GB/s, i.e. **19x** over the explicit path, and all of that gap is
compute.

So the ordering is: **WMMA first, then raise G**. Raising G without it is
measurably counterproductive. The fragment layouts, the lane-replication rule and
the gfx1151 rocWMMA patch in this repo are the starting point; the one open
question is precision, since WMMA takes FP16 inputs and q would be demoted (K is
already FP16). The standard recovery is splitting q into hi/lo FP16 halves and
issuing two WMMA passes on the QK side.

---

## Benchmark harness note

A config whose LDS request exceeds 64 KB **fails to launch silently** and then
looks infinitely fast. During this work a broken variant won an autotuning sweep
with a fake 453x speedup at a constant 2.6 us; only a correctness column
(max abs err jumping from 6e-08 to 0.3) caught it. Any sweep here should reject
over-budget configs up front *and* check `hipGetLastError()` after every launch —
`autotune.py` and `benchmark_optimizations.py` are exposed to the same failure.

---

## Repo note

`flash_attention.hpp` was removed in `5523ca2` ("perf: implement interleaved MMA +
prefetch pattern"), which does not mention it — 119 lines, recoverable at
`83511d1:flash_attention.hpp`. It is a scalar online-softmax forward pass rather
than a WMMA one, so nothing is blocked by its absence, but the deletion looks
unintentional.

---

## References

- [FlashInfer: accelerating self-attention for LLM serving](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) — operational intensity `H_qo/H_kv`, tensor cores for GQA/MQA decode
- [Attention Decode on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/gluon-attention-decode-mi450/README.html) — split-K, two WMMA ops, two-stage softmax, 85% of peak
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — `BLOCK_SIZE_M = 64`, seq-parallel tile scheduler
- [AMLA: MUL by ADD in FlashAttention Rescaling](https://arxiv.org/pdf/2509.25224) — cheaper online-softmax rescaling for MLA decode
- [WMMA on RDNA 3 (AMD GPUOpen)](https://gpuopen.com/learn/wmma_on_rdna3/) — note: `docs/wmma_fragment_layout_rdna3.md` in this repo corrects its A-fragment description (each lane loads a **row**)
