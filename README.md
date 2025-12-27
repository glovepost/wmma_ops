# rocWMMA Patch for PyTorch on gfx1151

**Optimized WMMA (Wave Matrix Multiply-Accumulate) operations for AMD gfx1151 (RDNA3.5 / Strix Halo) architecture**

Based on llama.cpp rocWMMA optimizations (PR #16827) and [Sébastien Vince's Deep Dive into Matrix Optimization on AMD GPUs](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html).

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Performance Results](#performance-results)
- [Building](#building)
- [Usage](#usage)
- [Testing](#testing)
- [Architecture Details](#architecture-details)
- [Kernel Variants](#kernel-variants)
- [Key Optimizations](#key-optimizations)
- [Optimization Techniques](#optimization-techniques)
- [Profiling and Analysis](#profiling-and-analysis)
- [Remaining Gap to rocBLAS](#remaining-gap-to-rocblas)
- [File Structure](#file-structure)
- [References](#references)

---

## Executive Summary

Successfully implemented and optimized a rocWMMA GEMM kernel for PyTorch targeting gfx1151, achieving **21.6 TFLOPS peak** (36% of theoretical 59.4 TFLOPS) with correct results across all test configurations.

This represents a **4× improvement** over the initial implementation (5.4 → 21.6 TFLOPS) through systematic optimization, reaching **53% of rocBLAS FP16** performance.

### Quick Stats

| Metric | Value |
|--------|-------|
| **Peak TFLOPS** | 21.6 (4096×4096) |
| **Utilization** | 36% of 59.4 TFLOPS peak |
| **vs rocBLAS FP16** | 53% (rocBLAS: ~41 TFLOPS) |
| **Improvement** | 4× over baseline |
| **Correctness** | ✅ All tests pass (rel_err < 1%) |

---

## Performance Results

### Final Benchmarks (Adaptive Tile Selection with K-Unrolling)

| Configuration | WMMA TFLOPS | % Peak | rocBLAS FP16 | % of rocBLAS | Status |
|---------------|-------------|--------|--------------|--------------|--------|
| 512×512×512 | 12.5 | 21.0% | 20.6 | 60% | ✅ |
| 1024×1024×1024 | 14.6 | 24.6% | 37.0 | 39% | ✅ |
| 2048×2048×2048 | 20.0 | 33.7% | 38.5 | 52% | ✅ |
| **4096×4096×4096** | **21.6** | **36.4%** | **41.0** | **53%** | ✅ |

**rocBLAS achieves 69% of peak (41/59.4 TFLOPS) while our kernel achieves 36% of peak.**

### Kernel Variant Comparison

#### Working Kernels (All Correctness Tests Pass)

| Kernel | Small (512) | Medium (1024) | Large (2048) | XLarge (4096) | Average | % of Peak |
|--------|-------------|---------------|--------------|---------------|---------|-----------|
| **Adaptive** | 10.13 | 15.98 | 20.30 | 21.17 | **16.89** | 28.4% |
| **Standard** | 14.05 | 11.16 | 20.08 | 21.88 | **16.79** | 28.3% |
| **K-Unroll** | 12.60 | 15.73 | 18.97 | 17.19 | **16.12** | 27.1% |
| **NoPrefetch** | 10.87 | 10.89 | 16.24 | 19.20 | **14.30** | 24.1% |
| **Hilbert** | - | - | - | 12.97 | - | 21.8% |
| **HighOcc** | 7.02 | 6.39 | 9.45 | 9.79 | **8.16** | 13.7% |

**Peak Theoretical**: 59.4 TFLOPS (gfx1151 FP16 WMMA)

#### Key Findings

1. **Adaptive kernel performs best overall** (16.89 TFLOPS avg)
   - Automatically selects optimal tile configuration
   - Best for medium (1024) and large (2048) matrices

2. **Standard kernel is most consistent** (16.79 TFLOPS avg)
   - Best for small (512) and very large (4096) matrices
   - Reliable performance across all sizes

3. **K-Unroll shows promise for medium matrices** (15.73 TFLOPS at 1024)
   - Reduces synchronization overhead
   - Currently integrated into adaptive selector

4. **HighOcc underperforms** (8.16 TFLOPS avg)
   - Lower register pressure doesn't compensate for reduced compute intensity
   - Not recommended for current workloads

### Correctness Tests

All correctness tests pass with acceptable FP16 precision:

| Test | Relative Error | Status |
|------|----------------|--------|
| 512×512×64 | < 0.1% | ✅ |
| 2048×2048×128 | < 0.1% | ✅ |
| 4096×4096×2048 | < 0.1% | ✅ |
| GEMM α=2.0, β=0.5 | < 0.001% | ✅ |
| GEMM in-place | < 0.001% | ✅ |

### Kernels with Known Issues

**Swizzled (XOR) Kernel**
- **Status**: ❌ FAIL (99.74% relative error)
- **Issue**: Fragment loading/storing pattern doesn't match WMMA layout requirements.
- **Symptom**: Extremely high TFLOPS (5000+) but garbage output.
- **Root Cause**: Almost certainly a **packing/ABI mismatch** between manual fragment construction and intrinsic requirements. Inverting the XOR swizzle only fixes the indexing; it does not guarantee the correct lane-to-register mapping for the `half16` ABI.
- **Recommendation**: Transition to vector-granularity swizzling (16B chunks) and use proven-good helper paths (`load_matrix_sync_lds`) for fragment construction.

**ASM-Opt Kernel**  
- **Status**: ❌ FAIL (40-54% relative error)
- **Issue**: Incorrect fragment loading pattern.
- **Root Cause**: Functional mismatch in manual register packing. While indexing may appear correct, the hardware expects a very specific bit-layout in the 8 VGPR pairs that manual `_Float16` loops often miss.
- **Fix Path**: Re-align with the helper-based fragment loading pipeline used by the standard kernel.

---

## XOR Swizzle Debugging Checklist

To avoid guesswork when debugging swizzle or fragment layout issues, follow this isolation sequence:

1. **Deterministic A/B Patterns**
   - `A[row,col] = row*256 + col`
   - `B[row,col] = row*256 + col + 0.25` (to distinguish from A)
2. **Roundtrip Kernel**
   - Write A into LDS using your swizzled addressing, read back using the inverse swizzle, and write to global.
   - If this fails, the inversion math is wrong (independent of WMMA).
3. **Fragment-Dump Kernel (No MMA)**
   - Load A operand fragment exactly as the MMA kernel would.
   - Write fragment data to global in canonical `[tile_row][tile_col]` view.
   - Catches lane replication or "which lane owns which col/row" mismatches.
4. **Epilogue-Dump Kernel (No Operand Loads)**
   - Initialize accumulator registers to a known pattern (e.g., `acc[i] = i + lane*0.01`).
   - Run epilogue store and validate the resulting matrix against expected row/col mapping.
   - If this fails, your store mapping is wrong.

### Layout Probe Test (Guardrail)

To institutionalize correctness, use a **Deterministic Layout Probe**:
- Use A/B patterns where the result C tile has a distinct structure (e.g., 0/120/240 row ramps).
- This catches transposes, lane mapping errors, and packing mismatches that subtle floating-point errors might hide.
- **CI Gating**: Every kernel variant (including adaptive selections) must pass this probe at small sizes before benchmarking.

## Building

### Using Docker (Recommended)

The project includes a Docker environment with ROCm 7.10, PyTorch, and all dependencies pre-configured for gfx1151.

**1. Build the Docker image:**

```bash
cd /path/to/wmma_ops
docker compose -f docker/docker-compose.benchmark.yml build
```

**2. Run the build and test suite:**

```bash
# Create .env file if it doesn't exist (required by docker-compose)
touch .env

# Build and test
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark \
  bash -c "export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:\$LD_LIBRARY_PATH && \
           cd /workspace/wmma_ops && ./build_and_test.sh"
```

**3. Interactive development:**

```bash
# Start an interactive shell in the container
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark bash

# Inside the container:
export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
cd /workspace/wmma_ops
pip install -e . --no-build-isolation
python test_rocwmma_patch.py
```

### Using pip (Host Installation)

If you have ROCm and PyTorch installed on your host system:

```bash
cd /path/to/wmma_ops
pip install -e . --no-build-isolation
```

### Build Requirements

- **Toolkit**: ROCm 7.9 or 7.10-preview
- **Compiler**: HIP compiler (`hipcc`)
- **Python**: PyTorch with ROCm support
- **Target**: gfx1151 (RDNA3.5 / Strix Halo)

---

## Usage

### Simple Matrix Multiply (C = A × B)

```python
import torch
import wmma_ops

# Create test tensors (FP16 input)
A = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)
B = torch.randn(2048, 4096, device='cuda', dtype=torch.float16)

# Use optimized WMMA matmul (FP32 output)
C = wmma_ops.matmul(A, B)

# Alternative: specify tile configuration
C = wmma_ops.matmul_tiled(A, B, 1)  # 1 = 128×64 tile

# Verify correctness
C_ref = torch.matmul(A, B)
print(f"Max error: {(C - C_ref).abs().max().item()}")
```

### Available Functions

#### Matrix Multiply Functions

| Function | Description |
|----------|-------------|
| `wmma_ops.matmul(A, B)` | Standard optimized kernel (recommended) |
| `wmma_ops.matmul_adaptive(A, B)` | Auto-selects optimal tile configuration |
| `wmma_ops.matmul_tiled(A, B, config)` | Explicit tile configuration (0-3) |
| `wmma_ops.matmul_kunroll(A, B)` | K-unrolled variant (2× fewer syncs) |
| `wmma_ops.matmul_noPrefetch(A, B)` | Without register prefetch |
| `wmma_ops.matmul_highOcc(A, B)` | High-occupancy variant |
| `wmma_ops.matmul_quad(A, B)` | Quad-buffered variant |
| `wmma_ops.matmul_native(A, B)` | gfx1151-specific with explicit intrinsics |

#### BLAS-Style GEMM (C = α × A × B + β × C)

| Function | Description |
|----------|-------------|
| `wmma_ops.gemm(A, B, alpha=1.0, beta=0.0, C=None)` | Standard GEMM with fused scaling |
| `wmma_ops.gemm_adaptive(A, B, alpha=1.0, beta=0.0, C=None)` | Auto-tuned GEMM with scaling |
| `wmma_ops.gemm_inplace(A, B, C, alpha=1.0, beta=0.0)` | In-place GEMM (modifies C directly) |

**Usage Example (GEMM with scaling):**
```python
import wmma_ops

# C = 2.0 * (A @ B) + 0.5 * C_prev
C = wmma_ops.gemm(A, B, alpha=2.0, beta=0.5, C=C_prev)

# In-place: C = 1.5 * (A @ B) + 0.3 * C (modifies C directly)
wmma_ops.gemm_inplace(A, B, C, alpha=1.5, beta=0.3)
```

### Recommendations

- **For Production Use**: Use `matmul_adaptive` - best overall performance
- **For Specific Sizes**: 
  - Small (512): Use `matmul` (Standard)
  - Medium (1024): Use `matmul_adaptive` (selects K-Unroll)
  - Large (2048+): Use `matmul_adaptive` (selects Standard 128×64)

---

## Testing

### Run Test Suite (Docker)

```bash
# Full build + test
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark \
  bash -c "export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:\$LD_LIBRARY_PATH && \
           cd /workspace/wmma_ops && ./build_and_test.sh"

# Or run tests only (after building)
docker compose -f docker/docker-compose.benchmark.yml run --rm benchmark \
  bash -c "export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:\$LD_LIBRARY_PATH && \
           cd /workspace/wmma_ops && python test_rocwmma_patch.py"
```

### Run Test Suite (Host)

```bash
python test_rocwmma_patch.py
```

### Run Profiling

```bash
python rocprof_wmma.py
```

### Run Benchmarks

```bash
# Quick benchmark (no Optuna required)
python autotune.py --quick

# Full Optuna tuning (requires: pip install optuna)
python autotune.py --trials 20

# Tune specific size
python autotune.py --size 4096 4096 2048
```

### Benchmark Methodology

- **Hardware**: AMD gfx1151 (Strix Halo / RDNA3.5)
- **Toolkit**: ROCm 7.9 or 7.10-preview
- **Warmup**: 3 iterations
- **Benchmark**: 20 iterations
- **Correctness**: Compared against PyTorch FP32 matmul
- **Tolerance**: < 1% relative error for correctness pass

---

## Architecture Details

### gfx1151 (RDNA3.5 / Strix Halo) Specifications

| Component | Specification |
|-----------|---------------|
| **Wavefront Size** | 32 threads (Wave32) |
| **SIMD Units per CU** | 2 × SIMD32 (dual-issue capable) |
| **Compute Units** | 40 CUs |
| **Peak Clock** | 2.9 GHz |
| **Peak TFLOPS** | 59.4 (FP16 WMMA) |
| **LDS** | 64 KB per CU |
| **VGPR File** | 192 KB per SIMD (1.5× larger than mobile RDNA3) |
| **Memory** | LPDDR5X (~256 GB/s) |

### WMMA Instruction

| Parameter | Value |
|-----------|-------|
| **Instruction** | `v_wmma_f32_16x16x16_f16` |
| **Tile Size** | 16×16×16 (M×N×K) |
| **Input Type** | FP16 (`half16`) |
| **Accumulator Type** | FP32 (`float8`) |
| **Wave Size** | 32 (w32 suffix) |
| **Latency** | ~32 cycles |

### WMMA Fragment Layout

For detailed information on fragment layouts, see [docs/wmma_fragment_layout_rdna3.md](docs/wmma_fragment_layout_rdna3.md).

**Key Points:**
- **RDNA3 WMMA requires lane replication**: lanes 0-15 and lanes 16-31 must contain identical data for A and B fragments
- **A Matrix**: Column-major format, each lane loads one column
- **B Matrix**: Transposed layout in LDS, each lane loads one row of transposed B
- **C/D Matrix**: Row-major format, lanes 0-15 cover even rows, lanes 16-31 cover odd rows

---

## Kernel Variants

### 1. Standard Kernel (Optimal)

**Configuration**: 128×64 tile, 4×2 warps, 2×2 register blocking

**Features**:
- Double buffering (overlaps loads with compute)
- GMEM spreading (register prefetch interleaved with MMA)
- Vectorized `half8` global loads
- Transposed B in LDS for `col_major` fragment access
- LDS padding (+8 halfs) to avoid bank conflicts

**Performance**: 21.9 TFLOPS (36.8% peak) at 4096×4096

### 2. Adaptive Kernel (Recommended)

**Configuration**: Auto-selects optimal variant based on matrix dimensions

**Selection Logic**:
- Small matrices (< 512): Uses 64×64 tiles
- Medium matrices (512-2048): Uses 128×64 tiles with K-unroll when beneficial
- Large matrices (> 2048): Uses 128×64 standard tiles

**Performance**: 21.6 TFLOPS (36.4% peak) at 4096×4096, best average across sizes

### 3. K-Unroll Kernel

**Configuration**: 128×64 tile with 2× K-unrolling

**Features**:
- Processes 2× BLOCK_K per iteration
- Reduces `__syncthreads` overhead by 50%
- Best for K dimensions in 768-1536 range

**Performance**: 17.2 TFLOPS (29% peak) at 4096×4096, but 15.7 TFLOPS at 1024×1024

### 4. No-Prefetch Kernel

**Simplification**: Removes register prefetch phase

**Trade-off**: Lower register pressure (~64 VGPRs) but less latency hiding

**Performance**: 19.2 TFLOPS (32% peak) at 4096×4096

### 5. High-Occupancy Kernel

**Configuration**: 64×32 tile, 4×1 warps, 2×1 register blocking

**Goal**: Maximize waves/CU by reducing VGPRs to ~50

**Result**: Worse performance (9.8 TFLOPS) — latency hiding more important than occupancy for this compute-bound workload

### 6. Native Kernel (gfx1151-specific)

**Configuration**: 128×64 tile, explicit inline assembly fences

**Features**:
- `lds_fence()`, `vmem_fence()`, `full_fence()` via inline asm
- Interleaved prefetch pattern (global loads between WMMA ops)
- `__builtin_prefetch` for software prefetch
- `amdgpu_waves_per_eu(4, 8)` occupancy hint

**Result**: ~same as adaptive (~20 TFLOPS) — async copy hardware not exposed in HIP

---

## Key Optimizations

### ✅ Implemented & Verified

| Optimization | Impact | Description |
|--------------|--------|-------------|
| **2×2 Register Blocking** | +80% | 4 WMMA tiles per warp (32×32 output) |
| **Double Buffering** | +20% | Ping-pong LDS buffers |
| **GMEM Spreading** | +15% | Prefetch into registers, interleaved with MMA |
| **Vectorized `half8` Loads** | +25% | 128-bit global loads |
| **128×64 Tile Shape** | +25% | Optimal A-matrix reuse |
| **Transposed B in LDS** | Required | Matches `col_major` fragment layout |
| **LDS Padding (+8 halfs)** | **+15-20%** | Eliminates bank conflicts (stride 24 vs 16) |
| **Pointer Increment** | +2% | Reduces VALU pressure in main loop |
| **amdgpu_waves_per_eu** | +2% | Compiler hint for occupancy targeting |
| **Epilogue Fusion (α/β)** | Saves 1 pass | Fused `C = αAB + βC` avoids separate scaling kernel |

### Low-Risk Perf Tweaks

- **Explicit Wave32**: Force/confirm wave32 compilation for gfx11 targets and assert at runtime using `__AMDGCN_WAVEFRONT_SIZE == 32`.
- **Compiler Hints**: Use `__restrict__` on A/B/C pointers and `__builtin_assume_aligned(ptr, 16)` on vectorized paths to encourage `global_load_b128` generation.
- **Wide LDS Reads**: Convert LDS fragment loads from 16 scalar half loads into two `ds_read_b128` + pack. This reduces LGKM overhead and bank conflict probability.

### ❌ Tested But Not Beneficial

| Optimization | Result | Reason |
|--------------|--------|--------|
| BLOCK_K=32 | Slower | Added loop overhead outweighed benefits |
| Odd LDS Stride (17, 33) | Slower | Forces scalar LDS access |
| Pre-transpose B on Host | Slower | Host overhead > kernel savings |
| Triple Buffering | Broken | Correctness issues with rotation |
| Inline Assembly Scheduling | No Change | Compiler scheduling already optimal |
| High-Occupancy Variant | Slower | Latency hiding > occupancy for this workload |

---

## Optimization Techniques

### LDS Bank Conflict Elimination

#### Current Approach: Padding

The current implementation uses **LDS padding** to avoid bank conflicts:

```cpp
#define LDS_PAD 8
constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 16 + 8 = 24 halfs = 48 bytes
```

**Problems with padding:**
1. **Wastes LDS memory**: 8 extra halfs per row = 16 bytes wasted per row
   - For BLOCK_M=128 rows: 128 × 16 = 2KB wasted per buffer
   - With double buffering: 4KB wasted total
2. **Doesn't guarantee conflict-free access**: Padding only helps if all accesses are sequential
3. **Breaks vectorized access alignment**: Stride of 24 halfs means rows aren't 128-bit aligned

#### Alternative: XOR-Based LDS Swizzle

XOR swizzle transforms memory indices so that bank conflicts are **mathematically impossible**:

```
Original index: (row, col)
Swizzled index: (row, col XOR f(row))
```

Where `f(row)` is chosen such that threads accessing different rows but same logical column will hit different banks.

**For BLOCK_K=16, KPACK=8:**
- K_GROUPS = 16 / 8 = 2
- Swizzle: `k_group_swizzled = k_group ^ (row & 1)`

**Memory Savings:**
- Padding approach: 18,432 bytes (with 2 buffers)
- XOR Swizzle: 12,288 bytes
- **Savings: 6,144 bytes (33%)**

**Status**: Implemented in `wmma_optimizations.hpp` but currently has correctness issues with fragment loading. Needs debugging before production use.

#### Critical Implementation Fixes for XOR Swizzle

When data is stored swizzled in LDS, fragment loading must account for the swizzle transformation. The following fixes are required:

**Fix 1: Fragment Loading with XOR Swizzle Inversion**

When loading fragments from swizzled LDS, you must invert the swizzle to get the correct data layout for WMMA:

```cpp
// INCORRECT: Direct access ignores swizzle
int frag_col = lane % 16;
for (int r = 0; r < 16; r++) {
    a0[r] = A_lds[curr][SwzA::to_flat(warp_m_base + r, frag_col)];
}

// CORRECT: Invert XOR swizzle
int frag_col_orig = lane % 16;  // Original column needed by WMMA

for (int r = 0; r < 16; r++) {
    int row = warp_m_base + r;
    
    // Invert XOR swizzle: find which swizzled column contains original column frag_col_orig
    int k_group_orig = frag_col_orig / 8;
    int k_local = frag_col_orig % 8;
    int k_group_swz = k_group_orig ^ (row & SwzA::K_GROUPS_MASK);
    int frag_col_swz = k_group_swz * 8 + k_local;
    
    a0[r] = *reinterpret_cast<const _Float16*>(&A_lds[curr][SwzA::to_flat(row, frag_col_swz)]);
}
```

**For B matrix** (similar fix with transposed layout):
```cpp
int frag_row_orig = lane % 16;  // Original row in transposed B layout

for (int kk = 0; kk < 16; kk++) {
    int n = warp_n_base + frag_row_orig;
    
    // Invert XOR swizzle for B
    int k_group_orig = kk / 8;
    int k_local = kk % 8;
    int k_group_swz = k_group_orig ^ (n & SwzB::K_GROUPS_MASK);
    int k_swz = k_group_swz * 8 + k_local;
    
    b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr][SwzB::to_flat(n, k_swz)]);
}
```

**Fix 2: Correct Epilogue Store Pattern**

WMMA fragment layout stores elements in a specific pattern. Each element `c_frag[i]` stores to row `i*2 + (lane/16)`, column `lane%16`:

```cpp
// INCORRECT: Wrong fragment layout assumption
int frag_row = lane % 16;
int frag_col_off = (lane / 16) * 8;
for (int e = 0; e < 8; e++) {
    int local_c = frag_col_off + e;
    C[gr0 * N + gc0] = c00[e];  // WRONG!
}

// CORRECT: Proper WMMA fragment layout
int frag_col = lane % 16;           // Column is fixed per lane
int frag_row_offset = lane / 16;    // 0 for lanes 0-15, 1 for lanes 16-31

for (int i = 0; i < 8; i++) {
    int frag_row = i * 2 + frag_row_offset;  // Rows: 0,2,4,...,14 or 1,3,5,...,15
    
    int gr0 = block_m + warp_m_base + frag_row;
    int gc0 = block_n + warp_n_base + frag_col;
    
    if (gr0 < M && gc0 < N) C[gr0 * N + gc0] = c00[i];
    
    // For 2×2 register blocking, handle all 4 tiles:
    int gc1 = gc0 + 16;  // Tile [0][1]
    if (gr0 < M && gc1 < N) C[gr0 * N + gc1] = c01[i];
    
    int gr1 = gr0 + 16;  // Tile [1][0]
    if (gr1 < M && gc0 < N) C[gr1 * N + gc0] = c10[i];
    
    if (gr1 < M && gc1 < N) C[gr1 * N + gc1] = c11[i];  // Tile [1][1]
}
```

**Complete Helper Function Example:**

```cpp
template<typename SwzA>
__device__ __forceinline__ void load_a_frag_swizzled(
    half16& a_frag,
    const __half* lds_base,
    int warp_m_base,
    int frag_col_orig
) {
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int row = warp_m_base + r;
        
        // Invert XOR swizzle
        int k_group_orig = frag_col_orig / 8;
        int k_local = frag_col_orig % 8;
        int k_group_swz = k_group_orig ^ (row & SwzA::K_GROUPS_MASK);
        int frag_col_swz = k_group_swz * 8 + k_local;
        
        a_frag[r] = *reinterpret_cast<const _Float16*>(&lds_base[SwzA::to_flat(row, frag_col_swz)]);
    }
}
```

**Alternative Approach**: If the swizzle unswizzling is too complex or has performance overhead, consider:
1. Store data swizzled (for bank conflict avoidance during global→LDS load)
2. Unswizzle during LDS→Fragment load into a temporary buffer
3. Load fragments from unswizzled buffer

This adds an extra LDS copy step but simplifies the fragment loading code.

**Testing Recommendations:**
1. Start with correctness: Test with small matrices (128×128) before performance
2. Compare against reference: Use non-swizzled kernel as reference
3. Verify swizzle math: Test XOR swizzle inversion logic separately
4. Check fragment layout: Verify fragment loading matches expected WMMA layout (see `docs/wmma_fragment_layout_rdna3.md`)
5. Profile bank conflicts: Use rocprof to verify XOR swizzle actually reduces conflicts

### L2 Cache Tile Rasterization

Simple row-major launch can cause L2 thrashing for large matrices. Column-major or chunked tile processing improves L2 cache locality.

**Expected Impact**: 5-15% improvement for large matrices (4096×4096+)

**Status**: Implemented in `wmma_optimizations.hpp` but not yet integrated into main kernels.

### Split-K for Skinny Matrices

Split-K assigns partial K-dimension slices to different work-groups, improving utilization for skinny matrices (small M or N, large K).

**Example**: M=16, N=4096, K=4096
- Without Split-K: Only 64 tiles → 60% utilization
- With Split-K factor 4: 256 tiles → 90% utilization

**Status**: Implemented in `wmma_optimizations.hpp` but not yet benchmarked.

### Register Pressure Management

**Current Register Usage**: Estimated ~91-92 VGPRs per thread (needs verification via `roc-obj-utils` or `rocprof`)

**To verify actual VGPR usage**:
```bash
# Extract from compiled kernel
roc-obj-utils --disassemble kernel.hsaco | grep -A 20 "COMPUTE_PGM_RSRC"
# Look for .vgprsnum value
```

**Compiler Hints**:
```cpp
__launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
// Note: amdgpu_num_vgpr does NOT work with templates in HIP
// Only amdgpu_waves_per_eu is supported with templates
```

**Important**: The `amdgpu_num_vgpr` attribute is **not supported with template kernels** in HIP. Use `amdgpu_waves_per_eu` to hint occupancy instead.

### SMEM-to-Register Double Buffering

Current kernel does **Global→LDS double buffering** but not **LDS→Register double buffering**. The problem is LDS loads and WMMA compute are serialized.

**Potential Improvement**: 5-10% by overlapping LDS loads with computation.

**Trade-off**: Doubles register usage for fragments (from ~128 to ~256 VGPRs), which may reduce occupancy.

---

## Profiling and Analysis

### Performance Characteristics

| K Dimension | Regime | Limiting Factor |
|-------------|--------|-----------------|
| K < 256 | Memory-bound | LPDDR5X bandwidth |
| K ≥ 512 | Compute-bound | WMMA throughput |

### Bottleneck Analysis (Compute-Bound Regime)

| Bottleneck | Contribution | Notes |
|------------|--------------|-------|
| `__syncthreads` overhead | ~20% | 128 barriers for K=2048 |
| Occupancy (5 waves/CU) | ~15% | 91 VGPRs limits to 5 waves |
| LDS bank conflicts (B scatter) | ~10% | Transpose pattern causes conflicts |
| WMMA pipeline bubbles | ~10% | Data dependencies within wave |

### Roofline Analysis

| Metric | Value |
|--------|-------|
| **Peak Compute** | 59.4 TFLOPS |
| **Peak Memory BW** | 256 GB/s |
| **Ridge Point** | ~106 ops/byte |
| **Our Intensity (K=2048)** | ~682 ops/byte |
| **Regime** | Compute-bound ✅ |

### ISA Analysis

#### Generated Assembly Inspection

| Component | Count | Status |
|-----------|-------|--------|
| **WMMA Instructions** | 4 | ✅ Correct (2×2 blocking) |
| **Global Loads (`global_load_b128`)** | 4 | ✅ Vectorized |
| **LDS Stores (`ds_store_b128`)** | 8 | ✅ Vectorized |
| **Dual-Issue (`v_dual_*`)** | 6-21 | ✅ Active |
| **Barriers (`s_waitcnt`)** | 10 | ⚠️ Necessary overhead |

#### WMMA Instruction Pattern

```assembly
s_waitcnt lgkmcnt(0)                                          ; Wait for LDS
v_wmma_f32_16x16x16_f16 v[1:8], v[57:64], v[49:56], v[1:8]    ; MMA 0
v_wmma_f32_16x16x16_f16 v[9:16], v[57:64], v[41:48], v[9:16]  ; MMA 1
v_wmma_f32_16x16x16_f16 v[17:24], v[33:40], v[49:56], v[17:24] ; MMA 2
v_wmma_f32_16x16x16_f16 v[25:32], v[33:40], v[41:48], v[25:32] ; MMA 3
```

**Key Finding**: Compiler groups WMMAs together intentionally. Attempts to interleave with inline assembly did not improve performance — the hardware scheduler handles dual-issue at runtime.

### Profiling Commands

**Profile LDS bank conflicts:**
```bash
rocprof --stats -o profile.csv -i metrics.txt ./your_kernel

# metrics.txt should include:
# LDSBankConflict
# LDSInstructions
# LDSBankConflictCycles

# View results
cat profile.csv | grep -E "LDSBankConflict|LDSInstructions"
```

**Interpretation**:
- `LDSBankConflict` / `LDSInstructions` = conflict rate (target: < 5%)
- High conflict rate indicates need for swizzling/padding

---

## Remaining Gap to rocBLAS

Our kernel achieves **53% of rocBLAS performance** (~21.6 vs ~41 TFLOPS). The gap is due to:

| Factor | Description |
|--------|-------------|
| **Hand-tuned Assembly** | rocBLAS uses offline-optimized ISA with perfect scheduling |
| **Adaptive Tile Selection** | rocBLAS selects optimal tile per matrix dimension |
| **Async Copy Hardware** | Uses hardware async LDS loads (not exposed in HIP) |
| **Register Allocation** | Compiler-level register allocation vs manual tuning |
| **Multi-kernel Fusion** | rocBLAS fuses alpha/beta scaling |

### Architecture-Specific Optimizations Explored

We attempted several gfx1151-specific optimizations without portability constraints:

| Optimization | Result | Notes |
|--------------|--------|-------|
| **Async global-to-LDS** | ❌ Not available | `__builtin_amdgcn_global_load_lds` not exposed in ROCm 7.x for gfx1151 |
| **Hardware prefetch** | ❌ Not available | `s_prefetch_data` instruction not supported on RDNA3.5 |
| **Explicit waitcnt** | ⚖️ No improvement | `lds_fence()`, `vmem_fence()` via inline asm perform same as compiler-managed |
| **Interleaved prefetch** | ⚖️ No improvement | WMMA ops (128 cycles) complete before global loads (400-800 cycles) |
| **Software prefetch** | ⚖️ Minimal impact | `__builtin_prefetch` adds ~1% improvement |

### Definitive Finding: No Async LDS Intrinsics for gfx1151

Based on extensive research of AMD ROCm docs, LLVM AMDGPU backend, and GPUOpen resources (as of late 2025):

- **No `__hip_ds_copy_async`** or similar hardware intrinsics exist for gfx1151
- **`llvm.amdgcn.load.to.lds`** exists but is synchronous (lowers to `global_load_lds` + `s_waitcnt`)
- **Async behavior** must be achieved through:
  - HIP runtime APIs (`hipMemcpyAsync` with streams) - host-device only
  - Manual overlap with `s_waitcnt vmcnt(x)` to allow in-flight loads - **already implemented**
  - Queue-level async (separate compute/copy queues) - not applicable for LDS

The only ISA pattern available for global-to-LDS:
```asm
buffer_load_dword v1, v0, s[sgpr0:sgpr3], ...  ; Global load
s_waitcnt vmcnt(0)                              ; Wait for load
ds_write_b32 v2, v1                             ; Write to LDS
s_waitcnt lgkmcnt(0)                            ; Wait for LDS visibility
```

GFX12 has `s_wait_dscnt` but still no async. ASYNC LDS and tensor ops are **not covered by the memory model** implemented by the AMDGPU backend; waits aren't inserted automatically and must be emitted explicitly.

### What Would Close the Gap

1. **AMD exposing true async LDS intrinsics** - requires hardware/firmware changes, not just software
2. **Write in AMDGCN assembly** directly with perfect scheduling (impractical, loses compiler optimizations)
3. **Use rocBLAS/hipBLASLt** for production (recommended - these use internal AMD optimizations)

### Expected Final Performance

The theoretical maximum without vendor-level assembly optimization is approximately **55-60% of peak** (~33-36 TFLOPS). Current optimizations in progress (XOR swizzle, L2 rasterization) could potentially reach **40-45% of peak** (~24-27 TFLOPS).

---

## Optimization Journey

### Evolution of Performance

| Version | TFLOPS | Change | Key Optimization |
|---------|--------|--------|------------------|
| Baseline | 5.4 | — | Basic WMMA implementation |
| + Multi-column blocks | 6.2 | +15% | BLOCK_N=64 for B reuse |
| + 2×2 Register Blocking | 9.7 | +56% | 4 accumulators per warp |
| + Vectorized Loads | 11.5 | +19% | `half8` global loads |
| + Double Buffering | 13.2 | +15% | Ping-pong LDS |
| + 128×64 Tile Shape | 16.5 | +25% | Increased A reuse |
| + GMEM Spreading | 17.4 | +5% | Register prefetch |
| + LDS Padding | **20.9** | **+20%** | Bank conflict elimination |

### Lessons Learned

1. **Tile shape matters more than expected**: 128×64 (tall) significantly outperforms 64×64 (square) due to A-matrix reuse
2. **Prefetching > Occupancy**: For compute-bound workloads, hiding latency via prefetch beats maximizing waves/CU
3. **Vectorized access is critical**: LPDDR5X heavily penalizes scalar loads
4. **Compiler scheduling is smart**: Inline assembly attempts didn't improve on LLVM's scheduling
5. **LDS transpose is required**: `col_major` B fragments need transposed data
6. **Odd strides hurt more than help**: Bank conflict avoidance via odd strides forces scalar access

---

## File Structure

```
wmma_ops/
├── README.md                        # This file
├── setup.py                         # Build configuration
├── wmma_gemm.hip                    # Main kernel implementation & pybind
├── wmma_kernels_optimized.hpp       # Optimized kernel variants (kunroll, quad, hilbert, etc.)
├── wmma_tile_mapping.hpp            # Hilbert curve tile mapping for L2 locality
├── wmma_xor_swizzle.hpp             # XOR swizzle, rasterization, Split-K
├── wmma_tile_selection.hpp          # Adaptive tile configuration
├── wmma_device_helpers.hpp          # Fragment loading helpers
├── rocwmma_patch/
│   └── rocwmma_gfx1151.hpp          # Custom rocWMMA patch header
├── docs/
│   └── WMMA_DEVELOPMENT_NOTES.md    # Consolidated development documentation
├── examples/                        # Reference implementations from other projects
├── autotune.py                      # Optuna-based auto-tuner
├── test_rocwmma_patch.py            # Test suite
├── benchmark_summary.py             # Benchmark utilities
└── build_in_docker.sh               # Docker build script
```

---

## References

### Primary Resources

1. **rocWMMA Documentation (ROCm)**
   - [rocWMMA Docs](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/index.html)
   - [API Reference](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html)

2. **Deep Dive into Matrix Optimization on AMD GPUs** (Sébastien Vince)
   - [Blog Post](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html)
   - Achievement: 49 TFLOPS on FP32 GEMM (60% faster than rocBLAS)

3. **AMD RDNA™ 3.5 ISA Reference Guide**
   - [AMD Documentation](https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture)

4. **LLVM AMDGPU Usage**
   - [LLVM Documentation](https://llvm.org/docs/AMDGPUUsage.html)
   - Code object metadata, register usage, ISA details

5. **rocBLAS Documentation (ROCm)**
   - [rocBLAS Docs](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html)

### Implementation References

- [llama.cpp PR #16827](https://github.com/ggml-org/llama.cpp/pull/16827) - Original optimizations
- [rocWMMA Library](https://github.com/ROCm/rocWMMA) - AMD's rocWMMA library
- [rocWMMA Samples](https://github.com/ROCm/rocWMMA/tree/develop/samples) - Reference kernels and usage patterns
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) - WMMA layout verification

---

## License

Based on rocWMMA library (MIT License) and llama.cpp optimizations.
