# wmma-ops

Optimized FP16-input, FP32-accumulation WMMA GEMM kernels for AMD Strix Halo
(`gfx1151`), exposed as a PyTorch extension and accompanied by standalone HIP
benchmarks. The ordinary-layout, FP32-output contract has a validated
4096-cubed peak of **41.322 TFLOPS**; a separate persistent block/K16-prepacked
FP16-output contract now sustains **49.035 TFLOPS**.

The project is a performance laboratory, not a drop-in replacement for
rocBLAS. Its useful outputs are the gfx1151 fragment helpers, a collection of
kernel schedules, and a reproducible method for deciding whether a result is
both correct and faster.

## Current status

The best fully validated standalone sample is **41.321799 TFLOPS**
(3.326064 ms) for a 4096 x 4096 x 4096 GEMM. The contract is FP16 A/B, FP32
accumulation, and FP32 output. Every one of the 16,777,216 output values was
checked against a rocBLAS FP32 reference.

The 41.322 number is a validated ordinary-layout peak, not a sustained record:
five fresh processes produced a 40.900 TFLOPS median and a 40.772-41.104
TFLOPS range. The separate prepacked contract is the current sustained
research leader, but it remains a distinct input-layout contract and is not
comparable to an ordinary-layout call that includes packing. The open research
gate is to close its 1.97% gap to 50 TFLOPS with five fresh processes and the
full numerical check in every process.

| Result | Shape | Numerical contract | Status |
|---|---|---|---|
| **41.322 TFLOPS** | 4096 cubed | FP16 inputs, FP32 accumulate/output | Validated standalone peak |
| **40.900 TFLOPS median** | 4096 cubed | Same | Five fresh processes; strict gate not met |
| **49.035 TFLOPS average** | 4096 cubed | block/K16-prepacked FP16 inputs/output | Five fresh 100-iteration processes; distinct persistent-input contract |
| **46.082 TFLOPS median** | 4096 cubed | FP16 inputs/output | Upstream `bench_half_half`; separate numerical contract |
| **85.907 INT4 TOPS** | 4096 cubed | prepacked linear W4A4, INT32 output | Exact full-output validation; separate numerical contract |
| **90.945 INT4 TOPS average** | 4096 cubed | paired-K prepacked linear W4A4, INT32 output | Five interleaved fresh pairs; exact; separate numerical contract |
| **110.229 INT4 TOPS** | IU4 issue-rate microbenchmark | signed INT4 inputs, INT32 accumulate | ISA qualification only; not an FP16 GEMM result |
| 21.6 TFLOPS | 4096 cubed | FP16 inputs, FP32 output | Historical PyTorch-extension result |
| about 41 TFLOPS | 4096 cubed | FP16 inputs/output | Historical `torch.mm` comparison; different contract |

The FP16-output comparison was reproduced from current
[`adelj88/rocm_wmma_gemm`](https://github.com/adelj88/rocm_wmma_gemm) commit
`281b5df` with ROCm 7.14. The exact published layout (A column-major, B
row-major, C row-major) measured 45.900, 46.082, and 46.212 TFLOPS in three
fresh processes, a 46.082 TFLOPS median and a 6.9-7.7% improvement over the
upstream 42.92 TFLOPS table entry. All 208 upstream same-precision tests passed.
This does not replace the project record: the output type differs, and the
upstream suite does not validate every output at the exact 4096-cubed benchmark
shape. See the performance ledger for the full command and timing distribution.

The separate block/K16-prepacked experiment now averages **49.035 TFLOPS**
across five fresh 100-iteration processes (48.980--49.116 TFLOPS, 2.802906 ms
mean median time) and reproduces the full 16,777,216-element rocBLAS error
tuple in every process. It combines progressive refill commit, scalar-offset
MUBUF addressing, and a measured two-VGPR register-phase shift. It assumes
persistent, block-packed A and B inputs, so it is not comparable to the
ordinary-layout FP32 record or to a call that includes packing. The previous
sustained leader was 48.614 TFLOPS, and the best isolated short sample remains
49.573 TFLOPS; isolated 50+ samples failed sustained same-pass checks and are
not promoted.

The clock-derived nominal ceiling is 59.4 TFLOPS: 20 WGPs x 1024
FLOP/WGP/cycle x 2.9 GHz. It is not a measured sustained ceiling.

The separate IU4 qualification uses `v_wmma_i32_16x16x16_iu4` and reaches
110.229 INT4 TOPS against its 118.8-TOPS clock-derived ceiling.  It validates
the nonuniform 16x16 product and the gfx1151 lane/register mapping exactly.
This is evidence about the integer matrix unit, not a new FP16 record: using it
for inference requires an INT4 activation/weight contract and scale handling.
The checked-in default uses eight independent chains; compile with
`-DIU4_CHAINS=16` to reproduce the peak configuration.  See
[`tools/bench_wmma_iu4.hip`](tools/bench_wmma_iu4.hip).

A complete prepacked linear-W4A4 GEMM built on that instruction reaches 85.907
INT4 TOPS at 4096 cubed and matches all 16,777,216 INT32 reference outputs
exactly.  The 128x128 block uses eight waves, eight independent accumulators
per wave, double-buffered 4 KiB LDS, and one barrier per K16.  This is a viable
new kernel architecture, but adopting it for model inference requires a new
quantization and quality contract.  See
[`tools/bench_wmma_iu4_gemm.hip`](tools/bench_wmma_iu4_gemm.hip).

The paired-K experimental variant keeps two K16 slices in four rotating LDS
slots and publishes the next pair once per two slices. It averages **90.945 INT4
TOPS** across five interleaved fresh candidate/control pairs (90.105--92.298
candidate; 84.484--84.969 one-slice control), with zero mismatches in every
process. It is compiled with `-DIU4_PAIR_K=1`; the default FP16 and one-slice
IU4 paths are unchanged.

Read [the performance ledger](docs/PERFORMANCE_STATUS.md) before comparing
numbers. It contains the exact schedule, distributions, rejected experiments,
resource metadata, and promotion rules.

## Documentation

| Document | Use it for |
|---|---|
| [Documentation index](docs/README.md) | Map of current, historical, and reference material |
| [Performance status](docs/PERFORMANCE_STATUS.md) | Current measurements, source audit, and next experiments |
| [Profiling guide](docs/PROFILING.md) | Timing, counters, ISA inspection, and shared-host discipline |
| [WMMA fragment layout](docs/wmma_fragment_layout_rdna3.md) | Verified gfx1151 lane/register mappings |
| [Annotated references](docs/wmma_references.md) | Primary AMD sources and architecture-porting hazards |
| [Decode-attention findings](docs/decode_attention_gfx1151.md) | Separate bandwidth-bound attention investigation |
| [Development notebook](docs/WMMA_DEVELOPMENT_NOTES.md) | Historical experiments; not authoritative current status |
| [RDNA3.5 ISA conversion](docs/rdna35_instruction_set_architecture.md) | Searchable conversion of AMD document 70649 |

## Requirements

- AMD Strix Halo / `gfx1151` with access to `/dev/kfd` and `/dev/dri`
- A gfx1151-capable ROCm toolchain
- Python 3 and a ROCm-enabled PyTorch build for the extension
- A C++20-capable AMD Clang/HIP compiler and rocBLAS development files for the
  standalone record harness

Current performance work uses ROCm 7.14. The checked-in benchmark and
profiling Dockerfiles are historical: they combine ROCm 7.9 Python packages
with ROCm 6.3 APT tooling. They can reproduce the old development environment,
but should not be described as a clean current stack or used to promote a new
record.

## Build the PyTorch extension

On a host with ROCm and PyTorch already installed:

```bash
export ROCM_PATH=/opt/rocm
python3 -m pip install -e . --no-build-isolation
python3 -c 'import wmma_ops; print(wmma_ops.__doc__)'
```

`setup.py` pins `PYTORCH_ROCM_ARCH=gfx1151`, invokes
`$ROCM_PATH/bin/hipcc`, and saves compiler intermediates for ISA inspection.
Record any non-default compiler flags with a performance result.

### Historical Docker environment

```bash
touch .env
docker compose -f docker/docker-compose.benchmark.yml build
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark \
  bash -lc 'cd /workspace/wmma_ops && ./build_and_test.sh'
```

The compose file requires `.env` because it was designed for model benchmarks
as well as this extension. No Hugging Face token is needed for the WMMA tests.

## Quick use

```python
import torch
import wmma_ops

a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)

# All matmul variants return FP32.
c = wmma_ops.matmul(a, b)
reference = a.float() @ b.float()

max_abs = (c - reference).abs().max().item()
normalized_max = max_abs / reference.abs().max().item()
print(max_abs, normalized_max)
```

Inputs must be two-dimensional FP16 tensors on the same GPU, with compatible
inner dimensions. Wrappers make inputs contiguous. WMMA-specific variants may
add alignment or K-multiple requirements and report them with `TORCH_CHECK`.

### Public extension bindings

The bindings fall into three groups. “Experimental” is intentional: a name is
not a claim that the variant is faster than `matmul`.

| Group | Bindings |
|---|---|
| Main paths | `matmul`, `matmul_adaptive`, `matmul_tiled` |
| Historical/experimental schedules | `matmul_hilbert`, `matmul_kunroll`, `matmul_native`, `matmul_zerocopy`, `matmul_quad`, `matmul_highOcc`, `matmul_noPrefetch`, `matmul_asmOpt`, `matmul_swizzled`, `matmul_xor_optimized`, `matmul_coop`, `matmul_pingpong`, `matmul_opt` |
| BLAS-style API | `gemm`, `gemm_inplace`, `gemm_adaptive` |

The checked-in extension does **not** expose Flash Attention. Attention notes
in this repository document a separate investigation, not a Python API.

BLAS-style use:

```python
c = wmma_ops.gemm(a, b, alpha=2.0)
c_previous = torch.zeros_like(c)
wmma_ops.gemm_inplace(a, b, c_previous, alpha=1.5, beta=0.3)
```

## Test the extension

```bash
./build_and_test.sh
python3 test_rocwmma_patch.py
python3 test_fragment_loading.py
```

`test_rocwmma_patch.py` is the main historical correctness/performance suite,
but its hard-coded variant list is not guaranteed to include every binding.
Check the list before relying on it for a new kernel. Use
`test_fragment_loading.py` whenever the gfx1151 register layout, LDS packing,
or output-store mapping changes.

Correctness for a record attempt means more than “the first few values look
right.” Compare the complete FP32 output against an FP32 reference and retain:

- finite/non-finite status;
- maximum absolute error;
- maximum error normalized by the maximum reference magnitude;
- RMS error.

The historical pass threshold is normalized maximum error below 1%.

## Reproduce the standalone 41.322 TFLOPS candidate

The standalone harness instantiates one schedule from the MIT-licensed
`adelj88/rocm_wmma_gemm` project. The build script verifies both its pinned
commit and the SHA-256 digest of its include tree.

```bash
git clone https://github.com/adelj88/rocm_wmma_gemm.git /tmp/rocm_wmma_gemm
git -C /tmp/rocm_wmma_gemm checkout \
  281b5dfd7fbff9cea80753bc55274a54f4a7c53a

tools/build_rocwmma_record.sh /tmp/rocm_wmma_gemm \
  build/rocwmma_record

# warm-ups, iterations per block, A/B/C element offsets, allocation mode
build/rocwmma_record 200 100 0 2048 0 combined-ab
```

The build fixes `gfx1151`, CU mode, `-O3`, fast math, and
`-amdgpu-unroll-threshold-local=700`. The harness uses deterministic
FP16-rounded inputs, five timing blocks, HIP event timing, a preallocated FP32
output, and a separate rocBLAS FP32 reference. It exits zero only if correctness
passes and the process median exceeds 41 TFLOPS.

One successful process is not enough to promote a record. Run the executable
from five fresh processes and report every process median and the total range.

## Benchmark the Python bindings

`benchmark_record.py` captures the environment, full-output correctness, raw
timing blocks, candidate order, and summary statistics as JSON:

```bash
for run_id in 1 2 3 4 5; do
  python3 benchmark_record.py \
    --warmup 10 --iterations 100 --blocks 5 \
    --run-id "$run_id" \
    --output "runs/record-$run_id.json"
done
```

The Python benchmark measures repeated extension calls, including the fresh
output allocation performed by each wrapper. Do not compare it directly with
the standalone harness without calling out that difference.

For quick exploratory tuning:

```bash
python3 autotune.py --quick
python3 autotune.py --size 4096 4096 4096
```

## Measurement rules

Performance claims in this repository follow these rules:

1. Use 4096 x 4096 x 4096 unless the result is explicitly labeled with another
   shape.
2. State input, accumulation, and output types. FP16-output rocBLAS is not a
   fair baseline for an FP32-output kernel.
3. Validate before timing and validate the complete output.
4. Keep counter collection out of timing runs; performance counters serialize
   dispatches on this stack.
5. Use at least ten warm-ups and 100 timed launches per block, five blocks per
   process, and five fresh processes for promotion.
6. Record the exact repository commit, compiler and ROCm versions, power/clock
   policy, matrix allocation layout, generated ISA, VGPR/SGPR/LDS use, and
   scratch use.
7. Promote a result only when every process passes correctness and every
   process median clears the active target.

See [the profiling guide](docs/PROFILING.md) for counter passes and ISA
inspection, and [the performance ledger](docs/PERFORMANCE_STATUS.md#record-protocol)
for the complete gate.

## Verified gfx1151 facts

- Wave size is 32 for this code.
- `v_wmma_f32_16x16x16_f16` performs 8192 FLOPs in 32 execution cycles and is
  rated at 1024 FLOP/WGP/cycle by AMD's Matrix Instruction Calculator.
- A and B fragments carry 16 elements per lane, replicated between lanes
  0-15 and 16-31.
- FP32 C/D fragments carry eight elements per lane; the two 16-lane halves
  hold alternating rows.
- RDNA3.5 LDS is 64 DWORD banks arranged as two sets of 32. For a wave32
  access, reason about `bank = (byte_address / 4) % 32`.
- RDNA4/gfx12 uses a different WMMA fragment layout. Do not compile its
  eight-elements-per-lane assumptions into gfx1151 code.

The authoritative mapping and citations live in
[the fragment-layout guide](docs/wmma_fragment_layout_rdna3.md) and
[the reference audit](docs/wmma_references.md).

## Repository layout

```text
wmma_gemm.hip                    PyTorch wrappers and extension bindings
wmma_kernel_largetile.hpp        Experimental large-tile implementation
wmma_kernel_variants.hpp         Historical kernel variants
wmma_tile_selection.hpp          Adaptive tile heuristic
wmma_device_helpers.hpp          Common gfx1151 helpers
wmma_xor_swizzle.hpp             XOR-LDS experiments
rocwmma_patch/                   gfx1151 fragment/intrinsic bridge
benchmark_record.py              Machine-readable extension benchmark
tools/rocwmma_record.hip         Standalone validated record harness
tools/build_rocwmma_record.sh    Pinned standalone build
test_*.py                        GPU correctness and integration checks
docs/                            Current guides, references, and history
examples/                        Vendored external examples and snapshots
docker/                          Historical benchmark/profiling environments
```

## Development guidance

- Treat `docs/PERFORMANCE_STATUS.md` as the current performance authority and
  `docs/WMMA_DEVELOPMENT_NOTES.md` as a historical notebook.
- Inspect generated code after changing a schedule. A plausible source-level
  optimization can silently increase VGPRs, lose occupancy, or spill.
- Keep architecture-specific fragment logic in the gfx1151 helper. RDNA4
  examples are not drop-in references.
- Add a new binding to the correctness and benchmark candidate lists in the
  same change.
- Preserve raw results for promoted claims; prose summaries alone are not
  enough to reproduce them.

## Origin and licensing

The code builds on ideas from llama.cpp rocWMMA work, AMD rocWMMA/Composable
Kernel sources, and Sébastien Vince's matrix-multiplication analysis. See
[the annotated references](docs/wmma_references.md) for exact sources and what
each one establishes.

The repository currently has no top-level license file. Do not infer a license
for the repository as a whole from the licenses of its upstream references.
The external kernel instantiated by the standalone harness is MIT-licensed at
its pinned upstream commit.
