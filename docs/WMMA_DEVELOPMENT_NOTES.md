# WMMA development notebook

> **Historical notebook:** This append-only file accumulated plans and
> measurements during several 2024-2025 sessions. It contains contradictory
> intermediate conclusions, superseded filenames, old environment assumptions,
> results from different matrix shapes, and hypotheses that were later
> rejected. It is intentionally retained as experiment history. Use
> [`PERFORMANCE_STATUS.md`](PERFORMANCE_STATUS.md) for the refreshed 2026-08-24
> source state and [`README.md`](README.md) for the documentation map. The
> current validated standalone peak is 41.322 TFLOPS; every “current status”
> label below is local to its historical section and is not the active plan.

The notebook begins with the original code-organization plan, then appends
fragment-layout investigations, correctness results, optimization experiments,
and performance summaries. Search by section title or date; it is not intended
to be read as one internally consistent specification.

## 2026-08-24: source-level raw-buffer prefetch boundary

The transposed 128x256 refill experiment was rebuilt from source with an
explicit raw-buffer descriptor and vector loads. This avoids the unsafe textual
conversion that previously produced an apparent >50-TFLOPS timing. The rebuilt
candidate was exact, but measured 45.897 TFLOPS against a 45.930-TFLOPS source
control. The same source path on 256x128 measured 47.307 versus 47.251 TFLOPS,
a noise-sized change. The temporary include override and binaries were deleted;
no patch was promoted. Descriptor-correct MUBUF alone does not remove the refill
bottleneck, so a future attempt must change the producer/consumer schedule.

The `warp_tile_m=2` fragment partition was rechecked with matching 128x128
launcher and kernel dimensions. Its original half-wave B loader was repaired
to cover the full tile, but the aligned candidate still hit an unspecified
launch failure before validation. This confirms a deeper fragment/loader
contract issue; no timing is recorded and no patch is retained.

An independent arithmetic experiment changed only the accumulator type: the
source used `v_wmma_f32_16x16x16_f16` and converted each output fragment to
FP16. It kept the 256x128, two-block/16-wave resource class and passed the full
reference tuple, but measured 47.922 TFLOPS. The FP32 WMMA path is correct but
slower than the retained FP16-accumulator schedule, so it is not promoted.

The WMMA issue order was then transposed to A-major: each A fragment visited
its four B fragments before advancing to the next A fragment. The source
candidate retained two-block/16-wave occupancy and the exact output tuple, but
measured 47.986 TFLOPS. This spacing change alone is slower than the retained
hand-scheduled phase-delta kernel and was not promoted.

## 2026-08-24: block traversal follow-up

The hand-scheduled delta-2 kernel was compared with the mapping layer's Morton
and CU-oriented modes. Modes 1/2/5/6 were exact but reached only 46.438,
46.463, 46.136, and 45.853 TFLOPS in the source screen, below the retained
leader. A fixed-4096 row-major override was also attempted in the hand image;
the loader rejected that manually shortened code object before launch. No
candidate or mapping claim was promoted, and all temporary artifacts were
removed after the lock was released.

The cyclic-bank idea was then rebuilt by composing two existing
register-aware pair swaps instead of globally remapping register text. This
produced a loader-valid three-bank cycle and passed exactness, but its required
five-process qualification measured 48.839, 48.698, 48.749, 48.649, and
48.712 TFLOPS (48.729 average, 48.649 floor). The isolated 49.059 result was
noise; the retained delta-2 image remains faster and more stable.

The opposite three-bank cycle was screened as well. It was loader-valid and
exact, but five processes measured 48.782, 48.721, 48.642, 48.722, and 48.733
TFLOPS (48.720 average, 48.642 floor). Both cycle directions are now closed;
future work should target synchronization or data movement rather than
accumulator placement.

The next physical-layout screen cyclically rotated three of the four FP16
accumulator banks while preserving the v0-based epilogue bank. Both rotations
were rejected by the gfx1151 loader before occupancy/correctness, whereas the
known pairwise bank swap still loads. This is recorded as an unsupported image
mapping, not as a performance result; the delta-2 leader remains unchanged.

The source block-prepacked kernel was also rebuilt with tighter wave-range
metadata (1--2, 2--2, and 1--4 waves per EU). Each candidate stayed exact and
reported the same two-block/16-wave occupancy, but measured 47.549, 47.622,
and 47.608 TFLOPS respectively. The scheduling attribute does not expose the
missing headroom and is closed as a standalone optimization.

The output epilogue was specialized for the divisible 4096x4096 benchmark so
each fragment stores directly without per-element bounds checks. It remained
exact at 47.850 TFLOPS with two resident blocks, which shows that tail
predicate overhead is not the missing 1.8%. The specialization was not kept.

# Code organization plan for `wmma_gemm.hip`

This document outlines logical chunks of code that can be extracted into separate header files to improve maintainability and organization.

## Current Structure

`wmma_gemm.hip` is ~2886 lines and contains:
- Device helper functions
- Multiple kernel implementations (10+ variants)
- Host wrapper functions
- Tile selection logic
- PyTorch bindings

## Proposed Header Files

### 1. `wmma_device_helpers.hpp` ⭐ HIGH PRIORITY
**Lines: ~35-75** (~40 lines)

**Contents:**
- `load_half8_to_lds()` - Vectorized LDS load helper
- `hw_prefetch_global()` - Software prefetch helper
- `lds_fence()`, `vmem_fence()`, `full_fence()` - Memory fence helpers

**Rationale:**
- Reusable across all kernels.
- Architecture-specific but independent.
- **Organization Tip**: Put only truly architecture-agnostic helpers here. Split architecture-specific bits behind a `gfxXXXX`-named header (e.g., `wmma_device_helpers_gfx1151.hpp`) to prevent this file from becoming a dumping ground.

**Dependencies:** `wmma_xor_swizzle.hpp` (for type definitions)

---

### 2. `wmma_tile_selection.hpp` ⭐ HIGH PRIORITY
**Lines: ~2080-2130** (~50 lines)

**Contents:**
- `enum class TileConfig` - Tile configuration enum
- `select_optimal_tile()` - Adaptive tile selection logic

**Rationale:**
- Standalone decision logic
- Used by multiple host wrappers
- Easy to test independently
- Could be enhanced without touching kernels

**Dependencies:** None (pure host-side logic)

---

**Rationale:**
- Standalone decision logic.
- Used by multiple host wrappers.
- Easy to test independently.

---

### 3. `wmma_torch_validate.hpp` ⭐ MEDIUM PRIORITY
**Lines: NEW**

**Contents:**
- `TORCH_CHECK` blocks, dtype/device/contiguity validation.
- Vectorization alignment checks.

**Rationale:**
- Host wrappers are tightly coupled to PyTorch.
- Extracting validation logic into its own header prevents "PyTorch swamps" and keeps launch wrappers near bindings for better readability.

---

### 4. `wmma_kernels_common.hpp` ⭐ MEDIUM PRIORITY
**Lines: Multiple kernel prologues** (~100-200 lines of repeated patterns)

**Contents:**
Common kernel setup code that's repeated:
- Constant definitions (WMMA_M, WMMA_N, WMMA_K, WARP_SIZE)
- Block/warp index calculations
- LDS buffer declarations
- Fragment initialization patterns
- Load index calculations

**Rationale:**
- Heavy duplication across kernels (~200+ lines repeated).
- Use `KernelConfig<>` for constants, index math, and static asserts.
- **Note**: Avoid over-templating too early. Keep index math and device helpers simple to avoid making SASS/ISA debugging miserable.

---

### 5. `wmma_kernels_standard.hpp` ⭐ LOW PRIORITY
**Lines: ~77-274, ~505-710** (~400 lines)

**Contents:**
- `wmma_gemm_kernel` - Standard optimized kernel
- `wmma_gemm_kernel_alphabeta` - Alpha/beta scaling variant
- Template instantiations

**Rationale:**
- Core production kernels
- Related functionality
- Could be separated for clarity

**Consideration:** These are the main kernels. Separation might not provide much benefit unless we're splitting by optimization strategy.

**Dependencies:** `wmma_device_helpers.hpp`, `wmma_xor_swizzle.hpp`, rocWMMA

---

### 6. `wmma_kernels_optimized.hpp` ⭐ LOW PRIORITY
**Lines: ~710-1520** (~800 lines)

**Contents:**
Optimization variant kernels:
- `wmma_gemm_kernel_kunroll` - K-unrolling variant
- `wmma_gemm_kernel_quad` - Quad-buffering variant
- `wmma_gemm_kernel_highOcc` - High-occupancy variant
- `wmma_gemm_kernel_noPrefetch` - No-prefetch variant
- `wmma_gemm_kernel_asmOpt` - Assembly-optimized variant

**Rationale:**
- Experimental/alternative kernels
- Could be separated for clarity
- Easier to enable/disable variants

**Consideration:** These are still used in production, so separation is mainly for organization.

**Dependencies:** Same as standard kernels

---

### 7. `wmma_kernels_specialized.hpp` ⭐ LOW PRIORITY
**Lines: ~1520-2080** (~560 lines)

**Contents:**
Specialized kernels:
- `wmma_gemm_kernel_gfx1151` - Architecture-specific kernel
- `wmma_gemm_kernel_zerocopy` - Zero-copy variant
- `wmma_gemm_kernel_native` - Native intrinsics variant
- `wmma_gemm_kernel_swizzled` - Swizzled LDS variant

**Rationale:**
- Special-purpose kernels
- Some may be experimental
- Clear separation of concerns

**Consideration:** Similar to optimized kernels - mainly for organization.

**Dependencies:** Same as standard kernels

---

## Recommended Extraction Order

### Phase 1: High-Value, Low-Risk Extractions

1. **`wmma_device_helpers.hpp`** ⭐⭐⭐
   - Clear separation
   - No dependencies on kernel internals
   - Immediate reuse benefit

2. **`wmma_tile_selection.hpp`** ⭐⭐⭐
   - Completely independent
   - Pure host-side logic
   - Easy to test

### Phase 2: Medium-Value Extractions

3. **Common validation helpers** (part of `wmma_host_helpers.hpp`)
   - Extract only helper functions, not full wrappers
   - Reduces code duplication
   - Minimal refactoring risk

4. **Kernel config templates** (part of `wmma_kernels_common.hpp`)
   - Extract only template helpers for common calculations
   - Reduces duplication without major restructuring

### Phase 3: Organizational (Optional)

5-7. Separate kernel files only if file becomes too large or for organizational clarity. Current ~2886 lines is manageable.

---

## Implementation Notes

### Include Strategy
- `wmma_xor_swizzle.hpp` should remain the base header with types and constants
- New headers should include `wmma_xor_swizzle.hpp` as needed
- `wmma_gemm.hip` includes all headers and provides PyTorch bindings

### Example Structure:
```cpp
// wmma_xor_swizzle.hpp (base)
- Types (half8, half16, float8)
- Constants (LDS_PAD)
- XOR swizzle implementations
- Rasterization utilities
- Split-K utilities

// wmma_device_helpers.hpp
#include "wmma_xor_swizzle.hpp"
- Device helper functions

// wmma_tile_selection.hpp
- TileConfig enum
- select_optimal_tile() function

// wmma_gemm.hip (main file)
#include "wmma_xor_swizzle.hpp"
#include "wmma_device_helpers.hpp"
#include "wmma_tile_selection.hpp"
- Kernel implementations
- Host wrappers
- PyTorch bindings
```

---

## Code Duplication Analysis

### High Duplication Areas:

1. **Tensor validation** (appears ~15+ times)
   - Same TORCH_CHECK blocks repeated
   - Could extract to helper functions

2. **Kernel constants setup** (appears ~10+ times)
   - WMMA_M, WMMA_N, WMMA_K, WARP_SIZE
   - Block/warp calculations
   - Could use template struct

3. **Tensor pointer extraction** (appears ~15+ times)
   - Same pattern: get pointers, check alignment
   - Could extract to helper

4. **Grid/block launch setup** (appears ~15+ times)
   - Similar patterns for dim3 grid/block
   - Could extract to helper

---

## Metrics

- **Current file size:** ~2886 lines
- **Estimated reduction after Phase 1:** ~90 lines extracted
- **Estimated reduction after Phase 2:** ~200-300 lines extracted (via helpers)
- **Estimated reduction after Phase 3:** ~1800 lines extracted (if splitting kernels)

**Recommendation:** Start with Phase 1 and Phase 2 to reduce duplication while maintaining clear structure. Phase 3 can be considered later if the file grows significantly.

# Analysis of Example Kernels for Potential Improvements

This document analyzes the example kernels in `examples/` to identify optimization techniques that could improve our WMMA kernel implementation.

## Key Findings

### 1. ✅ Hilbert Curve Tile Mapping (High Priority)

**Location**: `adelj_wmma_samples/hgemm/include/kernels/common.hpp` (lines 204-288)

**What it does**: 
- Maps linear block IDs to (row, col) coordinates using Hilbert curve space-filling pattern
- Improves L2 cache locality by processing spatially adjacent tiles together
- Uses optimized bit manipulation for GPU efficiency

**Implementation**:
```cpp
template<int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
    hilbert_tile_mapping(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
```

**Benefits**:
- Better L2 cache hit rates for large matrices
- Reduces memory bandwidth requirements
- Particularly effective for square matrices

**Current Status**: ❌ Not implemented in our kernels

**Recommendation**: ⭐⭐⭐ High priority - Easy to implement, could provide 5-10% improvement for large matrices

---

### 2. ✅ Register Prefetching (Medium Priority)

**Location**: `adelj_wmma_samples/hgemm/src/wmma_opt_3.cpp` (lines 154-228)

**What it does**:
- Prefetches next K-tile data to registers before storing to LDS
- Overlaps global memory loads with LDS stores from previous iteration
- Uses register buffers (`reg_buf`) to hold prefetched data

**Key Pattern**:
```cpp
// Prefetch to registers during computation
if (k_tile + 2 * block_k < K) {
    // Prefetch A tile to registers
    reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(next_A + ...);
}

// Later: Store registers to LDS
*dest_ptr = reg_buf[local_idx];
```

**Benefits**:
- Hides global memory latency
- Allows prefetching k+2 blocks ahead
- Reduces register pressure compared to LDS-only double buffering

**Current Status**: ❌ We use direct GMEM->LDS loads, not register prefetch

**Recommendation**: ⭐⭐ Medium priority - Could improve pipelining, but may increase register pressure

---

### 3. ✅ Cooperative Loading with Thread Division (Medium Priority)

**Location**: `adelj_wmma_samples/hgemm/src/wmma_opt_5.cpp` (lines 70-117)

**What it does**:
- Divides threads into two halves (`tid < half_block` vs `tid >= half_block`)
- First half loads A tiles, second half loads B tiles simultaneously
- Maximizes parallelism during LDS loading phase

**Key Pattern**:
```cpp
constexpr int half_block = num_threads / 2;
const int cid = tid % half_block;

if (tid < half_block) {
    // Load A tile
    *reinterpret_cast<vector_type*>(a_tiles_0 + dest_idx) = ...;
} else {
    // Load B tile
    *reinterpret_cast<vector_type*>(b_tiles_0 + dest_idx) = ...;
}
```

**Benefits**:
- Loads A and B in parallel
- Better thread utilization during load phase
- Reduces sync overhead

**Current Status**: ❌ We use all threads for both A and B sequentially

**Recommendation**: ⭐⭐ Medium priority - Good optimization, relatively easy to implement

---

### 4. ✅ Unified Shared Memory Buffer (Low Priority)

**Location**: Multiple files, e.g., `wmma_opt_5.cpp` (lines 24-33)

**What it does**:
- Uses single `__shared__ half lds_mem[2 * lds_size]` buffer
- Manually partitions into A and B regions using pointer arithmetic
- Eliminates separate buffer declarations

**Key Pattern**:
```cpp
__shared__ half lds_mem[2 * config_o5::lds_size];
half* a_tiles_0 = lds_mem;
half* b_tiles_0 = lds_mem + (block_m * block_k);
half* a_tiles_1 = lds_mem + lds_size;
half* b_tiles_1 = lds_mem + lds_size + (block_m * block_k);
```

**Benefits**:
- Slightly cleaner code organization
- No functional performance difference
- Easier to manage LDS size

**Current Status**: ✅ We use separate `A_lds[2][BLOCK_M][A_STRIDE]` and `B_lds[2][BLOCK_N][B_STRIDE]`

**Recommendation**: ⭐ Low priority - Cosmetic improvement only

---

### 5. ✅ Shared Memory Write Optimization (High Priority)

**Location**: `adelj_wmma_samples/hgemm/src/wmma_opt_5.cpp` (lines 227-310)

**What it does**:
- Uses shared memory as intermediate buffer for C matrix before writing to global
- Performs vectorized writes from LDS to global memory
- Processes output in chunks if tile doesn't fit in LDS

**Key Pattern**:
```cpp
#ifdef USE_SHARED_WRITE
// Step 1: Store WMMA fragments to shared memory
c_tile[row_local * block_n + col_local] = c_frags[wm][wn][i * 2];

__syncthreads();

// Step 2: Vectorized writes from shared memory to global memory
*reinterpret_cast<vector_type*>(C_base + ...) = 
    *reinterpret_cast<const vector_type*>(c_tile + ...);
#endif
```

**Benefits**:
- Coalesced global memory writes
- Vectorized write operations (float8)
- Better memory access patterns

**Current Status**: ❌ We write directly from fragments to global memory

**Recommendation**: ⭐⭐⭐ High priority - Could significantly improve write bandwidth for large N

---

### 6. ✅ Larger Block_K Size (Low-Medium Priority)

**Location**: `wmma_shared_warp_buf_vec.hpp` (line 43)

**What it does**:
- Some kernels use `block_k = 32` instead of `block_k = 16`
- Reduces number of sync points
- Increases LDS usage

**Trade-offs**:
- ✅ Fewer `__syncthreads()` calls
- ✅ Better K-tile reuse
- ❌ More LDS usage (may reduce occupancy)
- ❌ Larger prologue/epilogue

**Current Status**: ✅ We use `block_k = 16` (standard WMMA_K)

**Recommendation**: ⭐⭐ Medium priority - Worth testing, but may reduce occupancy on gfx1151

---

### 7. ✅ Vectorized Global Memory Writes (High Priority)

**Location**: `wmma_opt_5.cpp` (lines 299-310)

**What it does**:
- Uses `float8` (256-bit) vectorized writes from LDS to global memory
- Processes multiple elements per thread
- Handles boundary cases gracefully

**Key Pattern**:
```cpp
for(int i = tid * vector_width; i < (chunk_height * block_n); 
    i += num_threads * vector_width) {
    // Full vector write
    *reinterpret_cast<vector_type*>(C_base + ...) = 
        *reinterpret_cast<const vector_type*>(c_tile + ...);
}
```

**Benefits**:
- Maximizes memory bandwidth utilization
- Reduces write instruction count
- Better coalescing

**Current Status**: ❌ We write elements individually from fragments

**Recommendation**: ⭐⭐⭐ High priority - Should provide noticeable improvement

---

## Techniques Already Implemented

### ✅ Double Buffering
We already use double buffering with `A_lds[2][...]` and `B_lds[2][...]`

### ✅ Vectorized Global Loads
We use `half8` vectorized loads (matching their approach)

### ✅ LDS Padding
We use `LDS_PAD = 8` to avoid bank conflicts

### ✅ Register Blocking
We use 2x2 register blocking (4 accumulators per warp)

---

## Recommended Implementation Priority

### Phase 1: High-Impact, Low-Risk
1. **Hilbert Curve Tile Mapping** ⭐⭐⭐
   - Easy to add
   - Good cache locality improvement
   - Low risk of breaking existing code

2. **Shared Memory Write Optimization** ⭐⭐⭐
   - Significant write bandwidth improvement
   - Straightforward to implement
   - Clear performance benefit

3. **Vectorized Global Memory Writes** ⭐⭐⭐
   - Part of shared memory write optimization
   - High bandwidth improvement

### Phase 2: Medium-Impact, Medium-Complexity
4. **Cooperative Loading with Thread Division** ⭐⭐
   - Better thread utilization
   - Moderate implementation complexity

5. **Register Prefetching** ⭐⭐
   - Better latency hiding
   - Risk of register pressure increase
   - Needs careful tuning

### Phase 3: Lower Priority
6. **Larger Block_K Size** ⭐⭐
   - Needs benchmarking to verify benefit
   - May reduce occupancy

7. **Unified Shared Memory Buffer** ⭐
   - Cosmetic only
   - No performance benefit

---

## Implementation Notes

### Hilbert Curve Mapping
- The example uses a power-of-2 core + remainder approach for non-power-of-2 grids
- Fast path for perfect square power-of-2 grids
- Should integrate with our existing tile selection logic

### Shared Memory Write Optimization
- Need to handle chunking if output tile doesn't fit in LDS
- Requires careful synchronization between warps
- Should be optional (compile-time flag) for testing

### Cooperative Loading
- Need to ensure thread count is even (already true: 256 = 8*32)
- Should maintain our existing load index calculations
- Can combine with our current vectorized load approach

---

## Code References

- **Hilbert Mapping**: `examples/adelj_wmma_samples/hgemm/include/kernels/common.hpp:204-288`
- **Register Prefetch**: `examples/adelj_wmma_samples/hgemm/src/wmma_opt_3.cpp:154-228`
- **Cooperative Loading**: `examples/adelj_wmma_samples/hgemm/src/wmma_opt_5.cpp:70-117`
- **Shared Write**: `examples/adelj_wmma_samples/hgemm/src/wmma_opt_5.cpp:227-310`
- **ROCm WMMA Sample**: `examples/rocwmma_samples/perf_hgemm.cpp` (uses rocWMMA library, less relevant)

---

## Conclusion

The example kernels provide several optimization opportunities, with **Hilbert curve tile mapping** and **shared memory write optimization** being the highest-impact additions. These should be relatively straightforward to integrate and provide measurable performance improvements.

# Fragment Layout Analysis for RDNA3 WMMA

## Executive Summary

After analyzing the fragment layout documentation (`docs/wmma_fragment_layout_rdna3.md`) and comparing with the current implementation, I found that:

1. **For Padded LDS (current asmOpt)**: The current implementation **appears correct** based on the documentation pattern
2. **For XOR Swizzle LDS**: The proposed fix using `to_physical()` is **correct and necessary**
3. **Key Issue**: The asmOpt kernel uses **padded LDS**, so using `to_physical()` would be **incorrect** unless we switch to XOR swizzle

## Fragment Layout Requirements (from Documentation)

### A Matrix Fragment
From `docs/wmma_fragment_layout_rdna3.md`:
- **Lane L** loads column `(L % 16)` from the 16x16 A tile
- When loading from **row-major** source (which LDS is):
  - Each lane loads one **ROW** of A (from row-major storage)
  - Lane `lane` loads: `A_lds[row_offset + lane][k]` for `k = 0..15`
  - Code pattern:
    ```cpp
    const int lane = threadIdx.x % 16;
    const __half* row_ptr = A_lds + (row_offset + lane) * stride;
    for (int i = 0; i < 16; i++) {
        a_frag[i] = row_ptr[i];  // Load row 'lane', all 16 K values
    }
    ```

### B Matrix Fragment
- **Lane L** loads column `(L % 16)` from original B
- When B is **transposed** in LDS as `B_lds[N][K]`:
  - Each lane loads one **ROW** of transposed B (= one column of original B)
  - Lane `lane` loads: `B_lds[col_offset + lane][k]` for `k = 0..15`
  - Code pattern:
    ```cpp
    const int lane = threadIdx.x % 16;
    const __half* col_ptr = B_lds + (col_offset + lane) * stride;
    for (int k = 0; k < 16; k++) {
        b_frag[k] = col_ptr[k];  // Load row (col_offset + lane), all 16 K values
    }
    ```

## Current Implementation Analysis

### asmOpt Kernel (wmma_kernels_optimized.hpp:742-750)

**Current Code:**
```cpp
const int frag_col = lane_id % 16;

// Load A fragments
#pragma unroll
for (int row = 0; row < 16; row++) {
    a0[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + row][frag_col]);
    a1[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + 16 + row][frag_col]);
}

// Load B fragments
#pragma unroll
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + frag_col][kk]);
    b1[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + 16 + frag_col][kk]);
}
```

### The Problem

**This code is CORRECT for padded LDS layout!**

The current implementation:
1. ✅ For A: Iterates over rows (0..15), loads column `frag_col` from each row
   - This matches the pattern: `A_lds[row][frag_col]` for `row = 0..15`
   - This loads column `frag_col` across all 16 rows ✅

2. ✅ For B: Iterates over K values (0..15), loads from row `(warp_n_base + frag_col)`
   - This matches the pattern: `B_lds[warp_n_base + frag_col][kk]` for `kk = 0..15`
   - This loads row `(warp_n_base + frag_col)` across all 16 K values ✅

## The User's Concern

The user's proposed fix suggests using `to_physical()` for swizzled access. However, let's check if asmOpt uses swizzle or padding:

### LDS Layout in asmOpt Kernel

From `wmma_kernels_optimized.hpp:686-692`:
```cpp
constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 24
constexpr int B_STRIDE = BLOCK_K + LDS_PAD;  // 24

__shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
__shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
```

**Key Finding**: asmOpt uses **PADDED LDS** (LDS_PAD = 8), **NOT XOR swizzle**!

Therefore:
- ✅ Current direct `[row][col]` access is CORRECT
- ❌ Using `to_physical()` would be WRONG (that's for XOR swizzle only)

## Comparison with Other Kernels

### Standard Kernel (uses load_matrix_sync_lds)
Uses helper functions from `rocwmma_patch/rocwmma_gfx1151.hpp`:
- `load_matrix_sync_lds()` - handles row-major A loading
- `load_matrix_sync_lds_b_transposed()` - handles transposed B loading

These functions correctly implement the fragment layout.

### XOR Swizzle Kernel (wmma_xor_swizzle.hpp:922-949)
Uses `Swizzle::to_physical()` because it uses XOR swizzle (no padding):
```cpp
int phys0 = Swizzle::to_physical(row0, frag_col, A_STRIDE);
a0[r] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][phys0]);
```

This is correct for XOR-swizzled LDS.

## Critical Finding: Documentation vs Implementation Pattern Mismatch

After careful analysis, I found that the documentation shows TWO different loading patterns, which is confusing:

### Pattern 1: Load Column from Row-Major Source
```cpp
// Load column 'lane' from A (row-major source)
for (int i = 0; i < 16; i++) {
    a_frag[i] = A_lds[i][lane];  // Iterate rows, take column 'lane'
}
```

### Pattern 2: Load Row from Row-Major Source  
```cpp
// Each lane loads its own ROW
const __half* row_ptr = A_lds + (row_offset + lane) * stride;
for (int i = 0; i < 16; i++) {
    a_frag[i] = row_ptr[i];  // Load entire row
}
```

**These are different!** Pattern 1 loads a column (transposed access), Pattern 2 loads a row (direct access).

### Which is Correct?

The fragment register mapping shows: `a_frag[i] = A[i][lane%16]` - this means fragment position `i` should contain `A[row=i][col=lane]`.

If A_lds is stored as `[row][col]` (row-major), then:
- Pattern 1: `A_lds[i][lane]` → gets `A[row=i][col=lane]` ✅ CORRECT
- Pattern 2: `A_lds[lane][i]` → gets `A[row=lane][col=i]` ❌ WRONG (transposed)

**Pattern 1 (load column) is correct for the fragment layout!**

### Current asmOpt Implementation

Current code uses Pattern 1 (column loading):
```cpp
for (int row = 0; row < 16; row++) {
    a0[row] = A_lds[curr_buf][warp_m_base + row][frag_col];
}
```

This matches Pattern 1 and should be correct! ✅

## Conclusion

**The current asmOpt implementation matches the correct pattern from documentation.**

However, if the user is experiencing correctness issues, possible causes include:

### Possible Issues to Check:

1. **Lane Replication**: Does the code handle lanes 0-15 and 16-31 correctly?
   - Current code uses `frag_col = lane_id % 16`, which is correct ✅
   - But the fragment loading should ensure both half-waves get identical data

2. **Pointer Arithmetic**: The `&A_lds[curr_buf][warp_m_base + row][frag_col]` syntax
   - This should work correctly for 2D arrays
   - But if there are issues, explicit indexing might be clearer

3. **Data Type Conversion**: `_Float16` vs `__half`
   - Both should have same bit representation, but casting should be explicit

### Recommended Verification

1. ✅ Current code matches fragment layout spec for padded LDS
2. ⚠️ If using XOR swizzle, must use `to_physical()`
3. ⚠️ If correctness issues persist, check:
   - Lane replication (lanes 0-15 = lanes 16-31)
   - Boundary conditions
   - Store patterns (how data was written to LDS)

## Code Comparison

### What the Documentation Says (for row-major LDS):
```cpp
// A fragment loading
const int lane = threadIdx.x % 16;
const __half* row_ptr = A_lds + (row_offset + lane) * stride;
for (int i = 0; i < 16; i++) {
    a_frag[i] = row_ptr[i];  // Load entire row
}
```

### What Current Code Does:
```cpp
const int frag_col = lane_id % 16;  // Same as 'lane'
for (int row = 0; row < 16; row++) {
    a0[row] = A_lds[curr_buf][warp_m_base + row][frag_col];
}
```

**Wait!** There's a discrepancy:

The documentation says: **Load ROW `lane`** (all 16 K values)
Current code does: **Load COLUMN `frag_col`** (one value from each of 16 rows)

These are **opposite** approaches! Let me check which is correct for WMMA...

Actually, looking more carefully at the documentation:
- It says "Lane L loads column L%16" - but this refers to the **WMMA fragment layout**, not the load pattern
- When loading from **row-major** source, each lane loads a **row** of the source
- The hardware then interprets this as a column for the matrix multiply

So the documentation pattern loads `A_lds[row_offset + lane][k]` for `k=0..15` (a row).

But the current code loads `A_lds[warp_m_base + row][frag_col]` for `row=0..15` (a column).

**These are transposes of each other!**

However, both could be correct depending on how the data is organized. The key question is: what does the hardware expect?

From the fragment layout spec:
- Fragment register mapping shows: `a_frag[i] = A[i][lane%16]` (column-major for fragment)
- But when loading from row-major source, documentation shows loading a row

This suggests the hardware does the transpose internally. So both approaches might work, but they need to match the storage pattern.

Let me check how A_lds is written...

From the asmOpt kernel prologue:
```cpp
*A_lds[0][a_row][a_col] = ...;  // Stores at [row][col]
```

So A_lds is stored as `[row][col]` (row-major). 

If we want lane L to get column L%16, we should load:
- Row `warp_m_base + i` at column `frag_col` for `i = 0..15`
- OR: Row `warp_m_base + frag_col` at columns `0..15`

The current code does the first (column frag_col across rows), which should be correct if the hardware expects column-major fragments but we're storing row-major.

Actually, I think the confusion comes from mixing up:
1. How data is stored in LDS (row-major)
2. What the fragment layout means (column-major interpretation)
3. How to load it correctly

The safest interpretation: Follow the documentation pattern exactly - load a row from LDS for each lane. But the current code loads a column, which suggests either:
- The code is wrong, OR
- There's a transpose happening somewhere else

**Conclusion**: The user's concern is valid - there may be an issue with the fragment loading pattern. The documentation suggests loading rows, but current code loads columns. This needs verification with actual hardware testing or by comparing with working implementations.

# Fragment Layout Research Summary

## Research Question

User raised concern about fragment loading in `wmma_kernels_optimized.hpp` (asmOpt kernel), suggesting use of `to_physical()` for swizzle. Need to verify correctness against RDNA3 WMMA fragment layout specification.

## Key Documentation Reference

**File**: `docs/wmma_fragment_layout_rdna3.md`

### Fragment Layout Specification

#### A Matrix Fragment
- **Hardware expects**: `a_frag[i] = A[i][lane%16]` 
  - Fragment position `i` contains element `A[row=i][col=lane%16]`
  - This means: **each lane loads one column** across all 16 rows

- **Loading from row-major LDS** (`A_lds[row][col]`):
  ```cpp
  const int lane = threadIdx.x % 16;
  for (int i = 0; i < 16; i++) {
      a_frag[i] = A_lds[row_offset + i][lane];  // Load column 'lane'
  }
  ```
  - Iterate over rows (i=0..15)
  - Take column `lane` from each row
  - This loads **column** `lane` from the 16x16 tile

#### B Matrix Fragment
- **Hardware expects**: `b_frag[k] = B[k][lane%16]`
  - Fragment position `k` contains element `B[row=k][col=lane%16]`
  - This means: **each lane loads one column** of original B

- **Loading from transposed LDS** (`B_lds[N][K]`):
  ```cpp
  const int lane = threadIdx.x % 16;
  for (int k = 0; k < 16; k++) {
      b_frag[k] = B_lds[col_offset + lane][k];  // Load row 'lane' from transposed B
  }
  ```
  - Load row `(col_offset + lane)` from `B_lds[N][K]`
  - Iterate over K values (k=0..15)
  - This loads the row corresponding to column `lane` in original B

## Current asmOpt Implementation Analysis

### Code Location
`wmma_kernels_optimized.hpp:735-750` (asmOpt kernel)

### LDS Layout
```cpp
constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 24 (padded, NOT swizzled)
constexpr int B_STRIDE = BLOCK_K + LDS_PAD;  // 24

__shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
__shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
```

**Key Finding**: Uses **padded LDS**, NOT XOR swizzle.

### A Fragment Loading (lines 741-745)
```cpp
const int frag_col = lane_id % 16;

#pragma unroll
for (int row = 0; row < 16; row++) {
    a0[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + row][frag_col]);
    a1[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + 16 + row][frag_col]);
}
```

**Analysis**:
- ✅ Iterates over rows (0..15)
- ✅ Loads column `frag_col` from each row
- ✅ Pattern: `A_lds[row][frag_col]` for row=0..15
- ✅ This loads **column** `frag_col` → matches fragment layout spec ✅

### B Fragment Loading (lines 746-750)
```cpp
#pragma unroll
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + frag_col][kk]);
    b1[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + 16 + frag_col][kk]);
}
```

**Analysis**:
- ✅ Iterates over K dimension (0..15)
- ✅ Loads from row `(warp_n_base + frag_col)` of transposed B_lds
- ✅ Pattern: `B_lds[warp_n_base + frag_col][kk]` for kk=0..15
- ✅ This loads **row** `frag_col` from transposed B → matches fragment layout spec ✅

## Comparison with Helper Functions

### load_matrix_sync_lds (rocwmma_patch/rocwmma_gfx1151.hpp:186-204)

This function uses a **different pattern** - it loads a row:
```cpp
const int row = lane & 15;
const __half* row_ptr = base_ptr + row * ldm;
// Loads row_ptr[0..15] - which is a ROW, not a column
```

**Discrepancy**: Helper function loads rows, but documentation says to load columns.

**However**: When called as `load_matrix_sync_lds(a_frag, &A_lds[warp_m_base][0], stride)`, the `base_ptr` points to row `warp_m_base`, and then it adds `(lane%16) * stride`, so it's accessing row `(warp_m_base + lane%16)`. This is still a row, not a column.

**Resolution**: The helper functions may be using a different approach where the data is organized differently, OR there's a transpose happening. The important thing is that **the helper functions are used in working kernels**, so they must be correct for their usage pattern.

## Final Fragment Loading Verdict: Helper Path is Authoritative

Hardware validation confirms that the **standard kernel (helper-based)** is correct, while the **asmOpt (manual construction)** path is catastrophically wrong (100%+ relative error).

### The "Packing/ABI Mismatch" Gap
Correctness in WMMA is not just about loading the right 16 numbers; it's about matching the exact **packing and register layout** expected by the `__builtin_amdgcn_wmma_*` intrinsic.
- **Helper path**: Uses a proven-good vector-load + bitcast path.
- **Manual path**: Fails due to "right data, wrong lane/packing" mismatches between the manual `half16` construction and hardware expectations.

### Correctness Separation
1. **Indexing Correctness** (Logical to Physical): Correctness of padding vs. XOR swizzle to avoid bank conflicts.
2. **Fragment Packing Correctness** (Intrinsic ABI): Correctness of the data layout within the 8 VGPR pairs (packing order, lane replication).

**Crucial Fix**: To fix `asmOpt`, we must switch its fragment loading to use the same helper pipeline as the standard kernel, regaining correctness before re-introducing specific performance optimizations.

---

## PR Scoping Strategy

### [Phase 1] PR A: Safe Refactor + Correctness Rescue
- **Goal**: Extract headers and restore `asmOpt` correctness.
- **Scope**:
    - Extract Phase 1 headers (`wmma_device_helpers.hpp`, `wmma_tile_selection.hpp`).
    - Fix `asmOpt` by switching to helper-based fragment load + packing.
    - Add the **Layout Probe** test to verify `asmOpt` specifically.

### [Phase 2] PR B: Performance Additions (Flagged)
- **Goal**: Introduce high-impact optimizations safely.
- **Scope**:
    - Hilbert mapping (gated).
    - Shared-memory staging for C + vectorized stores (gated).
    - Benchmark results and layout probe validation for each flagged path.

## References

- `docs/wmma_fragment_layout_rdna3.md` - Fragment layout specification
- `rocwmma_patch/rocwmma_gfx1151.hpp` - Helper function implementations
- `wmma_xor_swizzle.hpp:922-949` - XOR swizzle kernel fragment loading (uses `to_physical()`)

# Fragment Loading Verification for RDNA3 WMMA

## Summary

After thorough analysis of the documentation and code, I've verified the fragment loading patterns. The key findings:

## Fragment Layout Specification (from docs/wmma_fragment_layout_rdna3.md)

### A Matrix Fragment
- **Hardware expects**: `a_frag[i] = A[i][lane%16]` (fragment position i contains A[row=i][col=lane])
- **From row-major LDS**: Each lane loads **column** `lane` across rows 0-15
- **Correct pattern**: 
  ```cpp
  for (int row = 0; row < 16; row++) {
      a_frag[row] = A_lds[row_offset + row][lane];
  }
  ```

### B Matrix Fragment  
- **Hardware expects**: `b_frag[k] = B[k][lane%16]` (fragment position k contains B[row=k][col=lane])
- **From transposed LDS**: Each lane loads **row** `lane` from B_lds[N][K]
- **Correct pattern**:
  ```cpp
  for (int k = 0; k < 16; k++) {
      b_frag[k] = B_lds[col_offset + lane][k];
  }
  ```

## Current asmOpt Implementation Verification

### A Matrix Loading (wmma_kernels_optimized.hpp:742-745)
```cpp
const int frag_col = lane_id % 16;
#pragma unroll
for (int row = 0; row < 16; row++) {
    a0[row] = A_lds[curr_buf][warp_m_base + row][frag_col];
}
```

**Analysis**:
- ✅ Iterates over rows (0-15)
- ✅ Loads column `frag_col` from each row
- ✅ Matches documentation pattern: `A_lds[row][frag_col]` for row=0..15
- ✅ This loads column `frag_col` → correct for fragment layout

**Verdict**: ✅ **CORRECT** for padded LDS

### B Matrix Loading (wmma_kernels_optimized.hpp:746-750)
```cpp
#pragma unroll
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = B_lds[curr_buf][warp_n_base + frag_col][kk];
}
```

**Analysis**:
- ✅ Iterates over K dimension (0-15)
- ✅ Loads from row `(warp_n_base + frag_col)` of transposed B_lds
- ✅ Matches documentation pattern: `B_lds[col_offset + lane][k]` for k=0..15
- ✅ This loads row `frag_col` from transposed B → correct for fragment layout

**Verdict**: ✅ **CORRECT** for padded LDS

## Comparison with Helper Functions

### load_matrix_sync_lds (rocwmma_patch/rocwmma_gfx1151.hpp:186-204)

```cpp
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds(
    fragment<matrix_a, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,
    int ldm
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int row = lane & 15;  // lane % 16
    const __half* row_ptr = base_ptr + row * ldm;
    
    // Direct load without swizzle accounting
    const half8_t v0 = *reinterpret_cast<const half8_t*>(row_ptr);
    const half8_t v1 = *reinterpret_cast<const half8_t*>(row_ptr + 8);
    ...
}
```

**Analysis**: This function loads a **ROW** from LDS (`row_ptr + row * ldm`), not a column!

**Wait - this contradicts the fragment layout spec!** Let me check if this is a different interpretation...

Actually, I realize the confusion: When `base_ptr` points to the start of a 16x16 tile, and we do `base_ptr + row * ldm`, we're getting row `row`. But the fragment layout needs column `lane`. 

Unless... the helper function assumes the data is already in the right format? Let me check how it's used.

### load_matrix_sync_lds_b_transposed (rocwmma_patch/rocwmma_gfx1151.hpp:206-223)

```cpp
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds_b_transposed(
    fragment<matrix_b, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,
    int col_stride
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int col = lane & 15;
    const __half* col_ptr = base_ptr + col * col_stride;
    
    const half8_t v0 = *reinterpret_cast<const half8_t*>(col_ptr);
    const half8_t v1 = *reinterpret_cast<const half8_t*>(col_ptr + 8);
    ...
}
```

**Analysis**: This loads column `col` (which equals `lane % 16`), iterating over K values via the pointer + offset. This matches the B fragment pattern.

## The Discrepancy

The helper function `load_matrix_sync_lds` loads a **row**, but the fragment layout documentation says to load a **column**. 

However, when we look at how the helper is called:
```cpp
load_matrix_sync_lds(a_frag[ti], &A_lds[curr_buf][warp_m_base + ti * WMMA_M][0], A_STRIDE);
```

The `base_ptr` is `&A_lds[warp_m_base + ti*16][0]` - pointing to the start of a specific row. But then inside the function, it adds `row * ldm` where `row = lane % 16`. So it's loading:
- `A_lds[warp_m_base + ti*16 + (lane%16)][0..15]` - which is row `(warp_m_base + ti*16 + lane%16)`

This is still loading a row, not a column!

**Unless** the helper function assumes the LDS is stored differently, or there's some transformation happening that I'm missing.

## Recommendation

Based on the fragment layout documentation, **the current asmOpt implementation appears correct**:
- Loads column `frag_col` for A fragments ✅
- Loads row `frag_col` for B fragments (from transposed B_lds) ✅

However, **the helper functions seem to use a different pattern** (loading rows for A). This discrepancy needs investigation.

### Possible Explanations

1. **The helper functions might be incorrect** - but they're used in working kernels, so this is unlikely
2. **There's a transpose happening elsewhere** - but I don't see evidence of this
3. **The documentation has an error** - possible but unlikely
4. **Both patterns work due to hardware interpretation** - possible if the hardware can handle both

### Suggested Next Steps

1. **Test the current asmOpt kernel** for correctness against reference implementation
2. **Compare with working kernels** that use `load_matrix_sync_lds` helpers
3. **If correctness issues exist**, verify:
   - How data is stored in LDS (prologue code)
   - Lane replication (lanes 0-15 = lanes 16-31)
   - Boundary conditions

### If Using XOR Swizzle

If switching asmOpt to use XOR swizzle (removing LDS_PAD), then the user's proposed fix using `to_physical()` would be correct:

```cpp
// With XOR swizzle, must use to_physical() to un-swizzle
for (int row = 0; row < 16; row++) {
    int phys_idx = Swizzle::to_physical(warp_m_base + row, frag_col, A_STRIDE);
    a0[row] = A_lds[curr_buf][phys_idx];
}
```

But currently, asmOpt uses padded LDS, so direct `[row][col]` access is correct.

# Fragment Loading Testing Summary

## Status: Test Script Created ✅

A comprehensive test script has been created to validate fragment loading correctness on real hardware. The test cannot be executed in this environment (PyTorch/ROCm not available), but it is ready to run when the proper environment is available.

## What Was Created

### 1. Test Script: `test_fragment_loading.py`

A comprehensive test that:
- Tests `asmOpt` kernel (`wmma_gemm_kernel_asmOpt`) against PyTorch reference
- Covers multiple matrix sizes (512x512 to 4096x4096)
- Provides detailed error statistics (absolute, relative, L2 errors)
- Compares with standard (known-working) kernel
- Includes small case analysis for debugging

### 2. Documentation: `TEST_FRAGMENT_LOADING.md`

Complete documentation explaining:
- How to run the test
- What to expect
- How to interpret results
- Current implementation analysis

## Current Implementation Status

### asmOpt Kernel (`wmma_kernels_optimized.hpp`)

**Implementation**: Uses **padded LDS** (not XOR swizzle)

```cpp
// A fragment loading (lines 742-745):
for (int row = 0; row < 16; row++) {
    a0[row] = A_lds[curr_buf][warp_m_base + row][frag_col];
    a1[row] = A_lds[curr_buf][warp_m_base + 16 + row][frag_col];
}

// B fragment loading (lines 747-750):
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = B_lds[curr_buf][warp_n_base + frag_col][kk];
    b1[kk] = B_lds[curr_buf][warp_n_base + 16 + frag_col][kk];
}
```

**Analysis**: Based on our research, this pattern should be **correct** for padded LDS:
- A: Loads column `frag_col` across rows 0-15 ✅
- B: Loads row `frag_col` (in transposed B_lds[N][K]) across all K values ✅

### XOR Swizzled Kernel (`wmma_xor_swizzle.hpp`)

**Implementation**: Uses **XOR swizzle** with `Swizzle::to_physical()`

```cpp
// B fragment loading (lines 939-949):
for (int kk = 0; kk < 16; kk++) {
    int n0 = warp_n_base + frag_col;
    int n1 = warp_n_base + 16 + frag_col;
    
    int phys0 = Swizzle::to_physical(n0, kk, B_STRIDE);
    int phys1 = Swizzle::to_physical(n1, kk, B_STRIDE);
    
    b0[kk] = B_lds[curr_buf][phys0];
    b1[kk] = B_lds[curr_buf][phys1];
}
```

**User's Reported Bug**: This pattern iterates over `kk` (K-dimension) for a single `n0/n1` (N-dimension), which may not match WMMA fragment layout requirements.

**User's Recommended Fix**: Use `n_row = lane_id % 16` instead of `frag_col` directly:

```cpp
const int n_row = lane_id % 16;
for (int kk = 0; kk < 16; kk++) {
    int n0 = warp_n_base + n_row;
    int n1 = warp_n_base + 16 + n_row;
    int phys0 = Swizzle::to_physical(n0, kk, B_STRIDE);
    int phys1 = Swizzle::to_physical(n1, kk, B_STRIDE);
    b0[kk] = B_lds[curr_buf][phys0];
    b1[kk] = B_lds[curr_buf][phys1];
}
```

## Expected Test Results

### Based on README.md Status:

- **ASM-Opt Kernel**: ❌ FAIL (40-54% relative error) - "Incorrect fragment loading pattern"
- **Swizzled (XOR) Kernel**: ❌ FAIL (99.74% relative error) - "Fragment loading/storing pattern doesn't match WMMA layout requirements"

### What the Test Will Reveal:

1. **If asmOpt passes** (max_rel_error < 1%):
   - Current implementation is correct for padded LDS
   - README status may be outdated
   - No changes needed to `wmma_kernels_optimized.hpp`

2. **If asmOpt fails** (max_rel_error > 1%):
   - Confirms correctness issue
   - Error patterns will indicate:
     - Systematic errors → wrong fragment layout
     - Random errors → precision issues (acceptable for FP16)
   - Need to investigate fragment loading logic

3. **Comparison with standard kernel**:
   - If asmOpt error >> standard kernel error → correctness bug
   - If asmOpt error ≈ standard kernel error → precision issue (acceptable)

## Running the Test

```bash
cd /path/to/wmma_ops

# Ensure extension is built
pip install -e . --no-build-isolation

# Run test
python3 test_fragment_loading.py
```

## Next Steps

1. **Run the test** on hardware with proper environment (PyTorch + ROCm)
2. **Analyze results**:
   - Check error magnitudes and patterns
   - Compare with standard kernel
   - Determine if changes are needed
3. **For XOR swizzled kernel**: Test separately (different test needed, or modify test script)

## Research Documents

- `FRAGMENT_LAYOUT_RESEARCH_SUMMARY.md`: Summary of fragment layout analysis
- `FRAGMENT_LAYOUT_ANALYSIS.md`: Detailed analysis
- `FRAGMENT_LOADING_VERIFICATION.md`: Verification of loading patterns
- `docs/wmma_fragment_layout_rdna3.md`: Technical reference

# Investigation: Why `rocwmma_patch/rocwmma_gfx1151.hpp` is in a Separate Folder

## Summary

The `rocwmma_patch/rocwmma_gfx1151.hpp` file is in a separate folder primarily for **organizational clarity**, but it is **NOT technically necessary**.

## What the File Does

The file contains a custom implementation that replaces/extends the standard ROCm rocWMMA library:

- Defines custom `rocwmma` namespace with fragment types, loaders, and WMMA intrinsics
- Optimized specifically for gfx1151 (RDNA3.5 / Strix Halo) architecture
- Implements XOR swizzle helpers for LDS bank conflict avoidance
- Provides wrappers around `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` intrinsic

## Current Setup

**setup.py includes both paths:**
```python
f'-I{rocm_path}/include/rocwmma',  # Standard ROCm library (included but not used)
f'-I{patch_dir}',                   # Custom patch directory (rocwmma_patch/)
```

**Usage in code:**
```cpp
#include "rocwmma_patch/rocwmma_gfx1151.hpp"  // Uses directory prefix in include
using namespace rocwmma;  // Uses the custom namespace from the patch
```

## Why a Separate Folder?

### ✅ Benefits

1. **Organizational Clarity**: Makes it immediately clear this is a "patch" or replacement implementation
2. **Documentation**: The folder name itself documents the intent
3. **Avoids Naming Conflicts**: Less likely to conflict with other headers in the main directory
4. **Separation of Concerns**: Keeps third-party/library code separate from application code

### ❌ Not Technically Required

The file could be moved to the main directory and work identically:

**Option 1: Move to main directory**
```cpp
// File: rocwmma_gfx1151.hpp (in wmma_ops/)
#include "rocwmma_gfx1151.hpp"
```

**Option 2: Keep current structure**
```cpp
// File: rocwmma_patch/rocwmma_gfx1151.hpp
#include "rocwmma_patch/rocwmma_gfx1151.hpp"
```

Both approaches work the same way - the include path in `setup.py` handles finding the file.

## Recommendation

**Keep the separate folder** for these reasons:

1. ✅ **Clear Intent**: The folder name documents that this is a custom patch/replacement
2. ✅ **Maintainability**: Future developers will immediately understand this replaces standard rocWMMA
3. ✅ **Consistency**: Follows common C++ practice of separating library/third-party code
4. ✅ **No Performance Impact**: Folder structure has zero impact on compilation or runtime

## Alternative: Simplify

If you want to simplify, you could:
1. Move `rocwmma_gfx1151.hpp` to the main `wmma_ops/` directory
2. Update includes from `#include "rocwmma_patch/rocwmma_gfx1151.hpp"` to `#include "rocwmma_gfx1151.hpp"`
3. Remove the `rocwmma_patch/` directory
4. Update `setup.py` to remove the patch_dir reference (already handled by `include_dirs`)

However, the current structure is **more maintainable and self-documenting**.

## Files That Use It

- `wmma_gemm.hip` - Main kernel implementation
- `wmma_kernels_optimized.hpp` - Optimization variant kernels

Both files use: `#include "rocwmma_patch/rocwmma_gfx1151.hpp"`

# Kernel Comparison: Standard (Working) vs asmOpt (Buggy)

## Executive Summary

**Key Finding**: The `asmOpt` kernel manually loads fragments using a **direct indexing approach**, while the **standard kernel uses helper functions** that correctly handle the fragment layout. The manual approach in `asmOpt` appears to be missing the correct lane-to-data mapping.

## Standard Kernel (CORRECT ✅)

### Fragment Loading Approach

The standard kernel uses helper functions from `rocwmma_patch/rocwmma_gfx1151.hpp`:

```cpp
// From wmma_gemm.hip:156-159
load_matrix_sync_lds(a_frag[ti], &A_lds[curr_buf][warp_m_base + ti * WMMA_M][0], A_STRIDE);
load_matrix_sync_lds_b_transposed(b_frag[tj], &B_lds[curr_buf][warp_n_base + tj * WMMA_N][0], B_STRIDE);
```

### load_matrix_sync_lds_b_transposed Implementation

```cpp
// From rocwmma_gfx1151.hpp:206-223
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds_b_transposed(
    fragment<matrix_b, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,   // points to B_lds[warp_n_base][0]
    int col_stride            // B_STRIDE = 24
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);  // lane = 0..31
    const int col = lane & 15;                       // col = 0..15 (lane % 16)
    const __half* col_ptr = base_ptr + col * col_stride;  // B_lds[warp_n_base + col][0]
    
    // Load 16 elements from this column (k=0..15)
    const half8_t v0 = *reinterpret_cast<const half8_t*>(col_ptr);      // k=0..7
    const half8_t v1 = *reinterpret_cast<const half8_t*>(col_ptr + 8);  // k=8..15
    
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
}
```

**Key Points**:
1. Each lane loads its own **column** (`col = lane % 16`)
2. Accesses `B_lds[warp_n_base + col][0..15]` via `col_ptr + offset`
3. Uses **stride** `col_stride` (24) to access the correct N-dimension index
4. Loads all 16 K values for that column

### load_matrix_sync_lds (A matrix) Implementation

```cpp
// From rocwmma_gfx1151.hpp:186-204
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds(
    fragment<matrix_a, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,   // points to A_lds[warp_m_base][0]
    int ldm                   // A_STRIDE = 24
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int row = lane & 15;                    // row = lane % 16
    const __half* row_ptr = base_ptr + row * ldm; // A_lds[warp_m_base + row][0]
    
    // Load 16 elements from this row (k=0..15)
    const half8_t v0 = *reinterpret_cast<const half8_t*>(row_ptr);      // k=0..7
    const half8_t v1 = *reinterpret_cast<const half8_t*>(row_ptr + 8);  // k=8..15
    
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
}
```

**Key Points**:
1. Each lane loads its own **row** (`row = lane % 16`)
2. Accesses `A_lds[warp_m_base + row][0..15]` via `row_ptr + offset`
3. Uses **stride** `ldm` (24) to access the correct M-dimension index
4. Loads all 16 K values for that row

## asmOpt Kernel (BUGGY ❌)

### Fragment Loading Approach

The asmOpt kernel manually loads fragments using direct indexing:

```cpp
// From wmma_kernels_optimized.hpp:741-750
const int frag_col = lane_id % 16;

// A fragment loading
#pragma unroll
for (int row = 0; row < 16; row++) {
    a0[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + row][frag_col]);
    a1[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + 16 + row][frag_col]);
}

// B fragment loading
#pragma unroll
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + frag_col][kk]);
    b1[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + 16 + frag_col][kk]);
}
```

**Key Points**:
1. Uses `frag_col = lane_id % 16` for **ALL lanes** (both 0-15 and 16-31)
2. For A: Iterates over **all 16 rows**, loads column `frag_col` from each
3. For B: Iterates over **all 16 K values**, loads from row `frag_col`

## Critical Difference Analysis

### Problem 1: Lane Replication Missing

**Standard Kernel**: Uses helper functions that handle lane replication internally
- The helper functions use `lane & 15` which naturally handles lanes 0-15 and 16-31 correctly
- The fragment structure ensures lanes 0-15 and 16-31 have the same data (required by RDNA3 WMMA)

**asmOpt Kernel**: All lanes load the same `frag_col = lane_id % 16`
- Lanes 0 and 16 both use `frag_col = 0`
- Lanes 1 and 17 both use `frag_col = 1`
- This creates the same data for lanes 0-15 and 16-31 ✅ (correct for replication)

**Verdict**: Lane replication appears correct in asmOpt ✅

### Problem 2: Fragment Loading Pattern

**Standard Kernel B Loading**:
```cpp
// For lane L (L % 16 = col):
col_ptr = base_ptr + col * col_stride  // = B_lds[warp_n_base + col][0]
// Loads: col_ptr[0..15] = B_lds[warp_n_base + col][0..15]
// Each lane L loads column col from B_lds
```

**asmOpt Kernel B Loading**:
```cpp
// For ALL lanes (using frag_col = lane_id % 16):
// Iterates kk = 0..15:
b0[kk] = B_lds[warp_n_base + frag_col][kk]
// This loads: B_lds[warp_n_base + frag_col][0..15]
// ALL lanes load from the SAME row (frag_col)
```

**WAIT!** This is the problem! 

The asmOpt kernel loads the **same row** for all lanes in a loop, but each lane should load a **different row**!

### Problem 3: Missing Per-Lane Data Distribution

**Standard Kernel**: 
- Each lane calls `load_matrix_sync_lds_b_transposed` independently
- Each lane loads its own column: `col = lane & 15`
- Lanes get different data based on their lane ID

**asmOpt Kernel**:
- All lanes execute the same loop
- All lanes load from the same `frag_col` (which is `lane_id % 16`)
- But the loop structure means all lanes are doing the same work!

Actually wait... let me reconsider. The loop in asmOpt is:
```cpp
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = B_lds[warp_n_base + frag_col][kk];
}
```

Since `frag_col = lane_id % 16`, different lanes will have different `frag_col` values, so they're loading different rows. This should be correct...

**BUT** - the standard kernel uses `col * col_stride` to access the data, while asmOpt uses direct `[warp_n_base + frag_col][kk]` indexing. These should be equivalent if the stride is handled correctly.

Let me check if the stride matters here... `col_stride = B_STRIDE = 24`, so:
- Standard: `B_lds[warp_n_base + col * 24 + kk]` (if treated as flat array)
- asmOpt: `B_lds[warp_n_base + frag_col][kk]` (2D array indexing)

Wait, but `B_lds` is declared as `__half B_lds[2][BLOCK_N][B_STRIDE]`, so 2D indexing should work correctly...

## The Real Issue: Loop Structure

Actually, I think I see it now! The asmOpt kernel is loading data **row-by-row** in a loop, but **all lanes execute this loop**. This means:

- Lane 0: loads `B_lds[warp_n_base + 0][0..15]` → correct
- Lane 1: loads `B_lds[warp_n_base + 1][0..15]` → correct
- ...
- Lane 15: loads `B_lds[warp_n_base + 15][0..15]` → correct

But the standard kernel loads **column-by-column** (via stride):
- Lane 0: loads `B_lds[warp_n_base + 0 * 24][0..15]` = `B_lds[warp_n_base][0..15]` → same as asmOpt
- Lane 1: loads `B_lds[warp_n_base + 1 * 24][0..15]` = `B_lds[warp_n_base + 24][0..15]` → **WAIT, this is different!**

Oh! I see the issue now. The standard kernel uses **stride-based access**:
- `base_ptr` points to `&B_lds[warp_n_base][0]`
- `col_ptr = base_ptr + col * col_stride` = `&B_lds[warp_n_base + col * 24]`
- But `B_lds` is declared as `[BLOCK_N][B_STRIDE]`, so this is accessing:
  - `B_lds[warp_n_base][col * 24 + kk]` which is **wrong** if interpreted as 2D!

Wait, let me check the declaration again... `__shared__ __half B_lds[2][BLOCK_N][B_STRIDE]`

So `B_lds[curr_buf][n][k]` means:
- First index: buffer (0 or 1)
- Second index: N dimension (0 to BLOCK_N-1)
- Third index: K dimension (0 to B_STRIDE-1)

So `B_lds[curr_buf][warp_n_base + n][k]` accesses:
- Row `warp_n_base + n` (N dimension)
- Column `k` (K dimension)

And `base_ptr = &B_lds[curr_buf][warp_n_base][0]` points to the start of row `warp_n_base`.

Then `col_ptr = base_ptr + col * col_stride` = `&B_lds[curr_buf][warp_n_base][col * 24]`...

But wait, that's accessing the **K dimension**, not the N dimension! This can't be right...

Unless... oh, I think I misunderstood. Let me look at how the standard kernel stores B:

```cpp
// From wmma_gemm.hip:129-134
if (b_valid && b_k < K) {
    half8 b_vec = *reinterpret_cast<const half8*>(B_base);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        B_lds[0][b_n + i][b_k] = reinterpret_cast<__half*>(&b_vec)[i];
    }
}
```

So B is stored as `B_lds[buf][n][k]` where:
- `n` is the N dimension (0 to BLOCK_N-1)
- `k` is the K dimension (0 to BLOCK_K-1 = 15)

And `load_matrix_sync_lds_b_transposed` is supposed to load from this transposed layout...

Actually, I think the issue is that `load_matrix_sync_lds_b_transposed` uses **pointer arithmetic with stride**, which works correctly for the 2D array layout. The stride `col_stride` is the stride in the **second dimension** (K dimension stride), but we're using it to step in the **first dimension** (N dimension).

Wait, that doesn't make sense either. Let me re-read the helper function...

Oh! I think I see it now. `base_ptr` points to `&B_lds[warp_n_base][0]`, and `col_ptr = base_ptr + col * col_stride` means we're adding `col * 24` to the pointer. Since `B_lds` is laid out as `[BLOCK_N][B_STRIDE]`, stepping by `col_stride` (24) bytes in the pointer space means we're stepping to the **next row** in the N dimension!

No wait, that's still not right. In C/C++, a 2D array `arr[N][M]` is laid out in row-major order, so:
- `arr[i][j]` is at offset `i * M + j` from the start
- So `&arr[i][j]` = `arr + i * M + j`

For `B_lds[BLOCK_N][B_STRIDE]`:
- `&B_lds[i][j]` = `B_lds + i * B_STRIDE + j`
- So `base_ptr + col * B_STRIDE` = `&B_lds[warp_n_base + 0][col * B_STRIDE]` which is still in the same row!

I think I'm confusing myself. Let me look at the actual usage more carefully...

Actually, I think the key insight is that **the standard kernel works correctly**, so the helper function must be doing the right thing. The difference must be in how asmOpt is loading the data.

Let me focus on the actual difference: the asmOpt kernel loads `B_lds[warp_n_base + frag_col][kk]` directly, which should be equivalent to what the standard kernel does. But the test shows it's wrong...

Maybe the issue is that asmOpt is missing the **fragment packing** that the helper functions do? Or maybe there's a difference in how the data is organized in memory?

Actually, I realize the real issue might be simpler: **the asmOpt kernel might be loading the wrong dimension or using the wrong indexing**. Let me check if `frag_col` should be used differently...

# Kernel Comparison Summary: Standard vs asmOpt

## Key Finding

The **standard kernel uses helper functions** that correctly handle fragment loading, while **asmOpt manually loads fragments** using a pattern that appears equivalent but produces incorrect results.

## Memory Access Pattern Comparison

### Standard Kernel (CORRECT ✅)

Uses `load_matrix_sync_lds_b_transposed()` helper function:

```cpp
// Helper function implementation:
const int lane = threadIdx.x & (WAVE_SIZE - 1);
const int col = lane & 15;  // col = lane % 16
const __half* col_ptr = base_ptr + col * col_stride;  // col_stride = 24
// base_ptr = &B_lds[warp_n_base][0]
// col_ptr = &B_lds[warp_n_base + col][0]

const half8_t v0 = *reinterpret_cast<const half8_t*>(col_ptr);      // k=0..7
const half8_t v1 = *reinterpret_cast<const half8_t*>(col_ptr + 8);  // k=8..15

// Unpack into fragment using bitcast_half()
for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
```

**Access Pattern**: `B_lds[warp_n_base + col][0..15]` where `col = lane % 16`

### asmOpt Kernel (INCORRECT ❌)

Manual loading:

```cpp
const int frag_col = lane_id % 16;
#pragma unroll
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + frag_col][kk]);
}
```

**Access Pattern**: `B_lds[warp_n_base + frag_col][0..15]` where `frag_col = lane_id % 16`

**Observation**: The memory access pattern appears equivalent!

## Critical Differences

### 1. Data Organization

- **Standard**: Uses `fragment<matrix_b, ...>` structure, then converts via `mma_sync()` wrapper
- **asmOpt**: Directly constructs `half16_t` arrays

### 2. Type Conversion

- **Standard**: Uses `bitcast_half()` function for type conversion
- **asmOpt**: Uses `reinterpret_cast<const _Float16*>()`

### 3. Vectorized vs Scalar Loading

- **Standard**: Loads `half8_t` vectors (8 elements at a time)
- **asmOpt**: Loads elements one at a time in a loop

### 4. Fragment Packing

The standard kernel's `mma_sync()` wrapper does additional packing:

```cpp
// From rocwmma_gfx1151.hpp:mma_sync()
half16_t a_vec, b_vec;
#pragma unroll
for (int i = 0; i < 16; i++) {
    a_vec[i] = bitcast_f16(a.x[i]);
    b_vec[i] = bitcast_f16(b.x[i]);
}
const float8_t r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, c_vec);
```

The asmOpt kernel constructs `half16_t` directly without going through the fragment structure.

## Hypothesis

The issue might be:

1. **Fragment structure organization**: The helper functions organize data in a specific way that matches WMMA requirements
2. **Type conversion**: `bitcast_half()` vs `reinterpret_cast` might handle edge cases differently
3. **Data alignment/packing**: Vectorized loads (`half8_t`) might ensure proper alignment

## Recommended Fix

Use the helper functions in asmOpt kernel:

```cpp
// Replace manual loading with helper functions:
fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[2];
load_matrix_sync_lds_b_transposed(b_frag[0], &B_lds[curr_buf][warp_n_base][0], B_STRIDE);
load_matrix_sync_lds_b_transposed(b_frag[1], &B_lds[curr_buf][warp_n_base + 16][0], B_STRIDE);

// Convert to half16_t for WMMA intrinsic (matching mma_sync wrapper):
half16_t b0, b1;
#pragma unroll
for (int i = 0; i < 16; i++) {
    b0[i] = bitcast_f16(b_frag[0].x[i]);
    b1[i] = bitcast_f16(b_frag[1].x[i]);
}
```

This ensures the same data organization and type conversion as the working standard kernel.

## Test Results Confirm

- Standard kernel: ✅ **0.000267 max error** (correct)
- asmOpt kernel: ❌ **162.62 max error** (609,013× larger, completely wrong)

This confirms there is a fundamental correctness issue in asmOpt's fragment loading that the helper functions correctly handle.

# Key Difference Between Standard and asmOpt Kernels

## The Critical Issue: Helper Function vs Manual Loading

After detailed comparison, I found the key architectural difference:

### Standard Kernel Uses Helper Functions

The standard kernel delegates fragment loading to helper functions in `rocwmma_patch/rocwmma_gfx1151.hpp`:

```cpp
load_matrix_sync_lds_b_transposed(b_frag[tj], &B_lds[curr_buf][warp_n_base + tj * WMMA_N][0], B_STRIDE);
```

These helper functions:
1. Handle the correct lane-to-data mapping
2. Properly pack data into the fragment structure
3. Use the correct memory access pattern with stride

### asmOpt Kernel Uses Manual Loading

The asmOpt kernel manually loads fragments:

```cpp
const int frag_col = lane_id % 16;
#pragma unroll
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = B_lds[curr_buf][warp_n_base + frag_col][kk];
    b1[kk] = B_lds[curr_buf][warp_n_base + 16 + frag_col][kk];
}
```

## The Problem: Fragment Organization

The helper function `load_matrix_sync_lds_b_transposed` uses **stride-based pointer arithmetic**:

```cpp
const __half* col_ptr = base_ptr + col * col_stride;  // col_stride = 24
const half8_t v0 = *reinterpret_cast<const half8_t*>(col_ptr);      // k=0..7
const half8_t v1 = *reinterpret_cast<const half8_t*>(col_ptr + 8);  // k=8..15
```

This accesses `B_lds[warp_n_base + col][0..15]` correctly.

But the **critical insight** is that the helper function uses **vectorized loads** (half8_t) and then **unpacks** them into the fragment structure in a specific way that matches the WMMA hardware requirements.

The asmOpt kernel loads elements **one at a time** in a loop, which might not match the fragment organization expected by the WMMA intrinsic.

## Solution: Use Helper Functions or Match Their Pattern Exactly

The asmOpt kernel should either:

1. **Use the helper functions** (simplest fix):
   ```cpp
   fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[2];
   load_matrix_sync_lds_b_transposed(b_frag[0], &B_lds[curr_buf][warp_n_base][0], B_STRIDE);
   load_matrix_sync_lds_b_transposed(b_frag[1], &B_lds[curr_buf][warp_n_base + 16][0], B_STRIDE);
   
   // Then convert to half16_t for WMMA intrinsic
   half16_t b0, b1;
   #pragma unroll
   for (int i = 0; i < 16; i++) {
       b0[i] = bitcast_f16(b_frag[0].x[i]);
       b1[i] = bitcast_f16(b_frag[1].x[i]);
   }
   ```

2. **Match the helper function's loading pattern exactly**:
   - Use stride-based pointer arithmetic
   - Use vectorized loads (half8_t)
   - Unpack in the same order

## Next Steps

1. Modify asmOpt to use helper functions for fragment loading
2. Test to confirm correctness
3. If performance is acceptable, keep the helper functions
4. If performance needs optimization, then optimize the helper function pattern

# Fragment Loading Correctness Test

## Purpose

This test validates the fragment loading logic in the `asmOpt` kernel (`wmma_gemm_kernel_asmOpt`) against PyTorch reference to verify correctness on real hardware.

Based on our fragment layout research, we want to confirm:
1. Whether the current implementation produces correct results
2. If there are correctness issues, what the error patterns are
3. How the asmOpt kernel compares to the standard (known-working) kernel

## Running the Test

```bash
cd /path/to/wmma_ops

# Build the extension (if not already built)
pip install -e . --no-build-isolation
# OR
./build_and_test.sh

# Run the test
python3 test_fragment_loading.py
```

## Expected Results

### Current Implementation Status (from README.md)

- **ASM-Opt Kernel**: ❌ FAIL (40-54% relative error)
- **Issue**: Incorrect fragment loading pattern
- **Fix Needed**: Correct fragment layout matching

### Test Coverage

The test covers:
1. **Multiple matrix sizes**: From small (512x512x64) to large (4096x4096x1024)
2. **Different aspect ratios**: Square, rectangular (M>N, N>M)
3. **Comparison with standard kernel**: Direct comparison with known-working implementation
4. **Small case analysis**: Detailed analysis of small matrices for debugging

### What to Look For

#### If Tests Pass (max_rel_error < 1%):
- ✅ Fragment loading is correct
- Current implementation matches expected behavior
- No changes needed

#### If Tests Fail (max_rel_error > 1%):
- ❌ Fragment loading has issues
- Check error patterns:
  - **Systematic errors**: Indicates wrong fragment layout/loading pattern
  - **Random errors**: Indicates precision/rounding issues (acceptable for FP16)
- Compare error magnitude:
  - If error >> standard kernel error → correctness bug
  - If error ≈ standard kernel error → precision issue (acceptable)

## Fragment Loading Analysis

### Current asmOpt Implementation

The `wmma_gemm_kernel_asmOpt` uses **padded LDS** (not XOR swizzle), so fragment loading is:

```cpp
// A fragment: load COLUMN frag_col (rows 0..15 of that column)
for (int row = 0; row < 16; row++) {
    a0[row] = A_lds[curr_buf][warp_m_base + row][frag_col];
    a1[row] = A_lds[curr_buf][warp_m_base + 16 + row][frag_col];
}

// B fragment: load ROW frag_col from B_lds[N][K] (all K values)
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = B_lds[curr_buf][warp_n_base + frag_col][kk];
    b1[kk] = B_lds[curr_buf][warp_n_base + 16 + frag_col][kk];
}
```

This pattern should be correct for padded LDS based on our research.

### User's Reported Bug

The user reported a bug with a **swizzled** version (uses `Swizzle::to_physical`):

```cpp
// INCORRECT (user's reported bug):
for (int kk = 0; kk < 16; kk++) {
    int n0 = warp_n_base + frag_col;
    int phys0 = Swizzle::to_physical(n0, kk, B_STRIDE);
    b0[kk] = B_lds[curr_buf][phys0];
}
```

This is in `wmma_xor_swizzle.hpp` (XOR swizzled kernel), not in `asmOpt`.

## Next Steps

1. **Run the test** on real hardware to get actual error measurements
2. **Analyze results**:
   - If asmOpt fails → investigate fragment loading in `wmma_kernels_optimized.hpp`
   - If asmOpt passes → current implementation is correct for padded LDS
3. **For XOR swizzled kernel** (`wmma_xor_swizzle.hpp`): Test separately as it uses different LDS layout

## Research Documents

- `FRAGMENT_LAYOUT_RESEARCH_SUMMARY.md`: Summary of fragment layout analysis
- `FRAGMENT_LAYOUT_ANALYSIS.md`: Detailed analysis of fragment layout
- `FRAGMENT_LOADING_VERIFICATION.md`: Verification of fragment loading patterns
- `docs/wmma_fragment_layout_rdna3.md`: Technical reference for RDNA3 WMMA fragment layout

# Fragment Loading Test Results - Hardware Validation

**Date**: 2024-12-27  
**Environment**: ROCm 7.9 Benchmark Docker Container  
**GPU**: gfx1151 (RDNA3.5)  
**Kernel Tested**: `wmma_gemm_kernel_asmOpt` (from `wmma_kernels_optimized.hpp`)

## Executive Summary

**❌ CRITICAL CORRECTNESS ISSUE CONFIRMED**

The `asmOpt` kernel has systematic correctness errors with:
- **Max relative error**: 117-146% (all test cases)
- **Error magnitude**: **609,013× larger** than standard kernel
- **Systematic nature**: 92-98% of elements have error > 1.0
- **Pattern**: All output values are incorrect, suggesting wrong fragment loading

## Test Results

### All Test Cases Failed ❌

| Test Case | Max Abs Error | Max Rel Error | % Elements Error > 1.0 |
|-----------|---------------|---------------|------------------------|
| Small (512×512×64) | 50.50 | 117.08% | 92.7% |
| Small (512×512×128) | 75.77 | 136.39% | 94.8% |
| Medium (1024×1024×256) | 106.34 | 124.56% | 96.3% |
| Large (2048×2048×512) | 162.62 | 138.79% | 97.4% |
| XL (4096×4096×1024) | 239.42 | 136.39% | 98.2% |
| Rectangular M>N | 156.72 | 136.05% | 97.4% |
| Rectangular N>M | 161.62 | 145.95% | 97.4% |

### Comparison with Standard Kernel

- **Standard kernel max error**: 0.000267 ✅
- **asmOpt kernel max error**: 162.624008 ❌
- **Error ratio**: **609,013× larger** (confirms correctness bug)

### Small Case Analysis (M=64, N=64, K=16)

**Reference output** (correct):
```
[[  0.   0.   0.   0.   0.   0.   0.   0.]
 [120. 120. 120. 120. 120. 120. 120. 120.]
 [240. 240. 240. 240. 240. 240. 240. 240.]
 [360. 360. 360. 360. 360. 360. 360. 360.]
 ...
```

**asmOpt output** (incorrect):
```
[[1240. 1240. 1240. 1240. 1240. 1240. 1240. 1240.]
 [1240. 1240. 1240. 1240. 1240. 1240. 1240. 1240.]
 [1240. 1240. 1240. 1240. 1240. 1240. 1240. 1240.]
 ...
```

**Key Observations**:
- All output values are identical (1240) - completely wrong pattern
- Reference shows increasing values per row (0, 120, 240, ...)
- This indicates the kernel is not computing the correct matrix multiplication

## Error Analysis

### Error Characteristics

1. **Systematic Errors**: 92-98% of elements have error > 1.0
   - This is NOT a precision issue (FP16 rounding would cause <1% errors)
   - Indicates **fundamental correctness bug** in fragment loading

2. **Error Pattern**:
   - Errors are consistent across all matrix sizes
   - Relative error ~130-140% (more than 100% means output is completely wrong)
   - All values in output are incorrect

3. **Comparison with Standard Kernel**:
   - Standard kernel: <0.001 relative error ✅ (correct)
   - asmOpt kernel: >100% relative error ❌ (completely wrong)

## Root Cause Analysis

### Current Implementation

The `asmOpt` kernel uses **padded LDS** (not XOR swizzle) and loads fragments as:

```cpp
// A fragment loading (wmma_kernels_optimized.hpp:742-745)
for (int row = 0; row < 16; row++) {
    a0[row] = A_lds[curr_buf][warp_m_base + row][frag_col];
    a1[row] = A_lds[curr_buf][warp_m_base + 16 + row][frag_col];
}

// B fragment loading (wmma_kernels_optimized.hpp:747-750)
for (int kk = 0; kk < 16; kk++) {
    b0[kk] = B_lds[curr_buf][warp_n_base + frag_col][kk];
    b1[kk] = B_lds[curr_buf][warp_n_base + 16 + frag_col][kk];
}
```

### Possible Issues

Based on the test results and fragment layout research:

1. **Fragment Layout Mismatch**: The fragment loading may not match RDNA3 WMMA requirements
2. **Lane Replication**: RDNA3 WMMA requires lanes 0-15 and 16-31 to have identical data for A and B fragments
3. **B Fragment Layout**: B fragments may need different organization than currently implemented

### Research Findings

From `FRAGMENT_LAYOUT_RESEARCH_SUMMARY.md`:
- RDNA3 WMMA requires specific fragment layout
- Each lane must hold correct data organization
- B fragments are conceptually transposed for WMMA

## Recommendations

1. **✅ CONFIRMED**: The `asmOpt` kernel has a correctness bug (hardware validated)
2. **Investigation Needed**: Review fragment loading logic against RDNA3 WMMA specification
3. **Compare with Standard Kernel**: The standard kernel works correctly - use it as reference
4. **Fix Priority**: HIGH - kernel produces completely wrong results

## Next Steps

1. Compare `asmOpt` fragment loading with standard kernel (working implementation)
2. Review fragment layout documentation (`docs/wmma_fragment_layout_rdna3.md`)
3. Apply fix based on user's recommended pattern (from bug report)
4. Re-test after fix

## Test Command

```bash
cd /path/to/wmma_ops
docker run --rm -v "$(pwd)":/workspace/wmma_ops -w /workspace/wmma_ops <rocm-container> \
  bash -lc "pip install -e . --no-build-isolation >/dev/null 2>&1 && \
  python3 test_fragment_loading.py"
```

# AMD RDNA3/gfx1151 WMMA Fragment Layout Reference

## Overview

This document describes the exact register and lane mapping for the `v_wmma_f32_16x16x16_f16` instruction on AMD RDNA3 architecture (gfx1100, gfx1101, gfx1102, gfx1151, etc.) in **wave32 mode**.

## Key Characteristics

| Property | Value |
|----------|-------|
| Wave Size | 32 threads (wave32) |
| Tile Size | 16x16x16 (M×N×K) |
| A/B Input | FP16 (packed, 2 per VGPR) |
| C/D Output | FP32 (1 per VGPR) |
| VGPRs for A | 8 per lane |
| VGPRs for B | 8 per lane |
| VGPRs for C/D | 8 per lane |

## Critical: Lane Replication Requirement

**RDNA3 WMMA requires that lanes 0-15 and lanes 16-31 contain IDENTICAL data for A and B fragments.**

This means:
- Lane 0 must have the same A/B data as Lane 16
- Lane 1 must have the same A/B data as Lane 17
- ... and so on up to Lane 15 = Lane 31

The hardware uses both half-waves but expects them to have replicated input data.

---

## A Matrix Fragment Layout

### Memory Layout
A is stored in **column-major** format for WMMA:
- `A[i][k]` is at memory offset `k * 16 + i`
- Each column of A (16 elements) goes into one lane's fragment

### Fragment Register Mapping

For **lane L** (where `effective_lane = L % 16`):

```
a_frag[0]  = A[0][effective_lane]   (packed with a_frag[1])
a_frag[1]  = A[1][effective_lane]
a_frag[2]  = A[2][effective_lane]   (packed with a_frag[3])
a_frag[3]  = A[3][effective_lane]
...
a_frag[14] = A[14][effective_lane]  (packed with a_frag[15])
a_frag[15] = A[15][effective_lane]
```

**Physical Register Layout (8 VGPRs per lane):**

| VGPR | Bits [15:0] | Bits [31:16] |
|------|-------------|--------------|
| v0 | A[0][lane%16] | A[1][lane%16] |
| v1 | A[2][lane%16] | A[3][lane%16] |
| v2 | A[4][lane%16] | A[5][lane%16] |
| v3 | A[6][lane%16] | A[7][lane%16] |
| v4 | A[8][lane%16] | A[9][lane%16] |
| v5 | A[10][lane%16] | A[11][lane%16] |
| v6 | A[12][lane%16] | A[13][lane%16] |
| v7 | A[14][lane%16] | A[15][lane%16] |

### Load Code Pattern (from row-major source)

```cpp
// A_src is [M][K] row-major in global memory
// A_lds is [M][K] row-major in LDS

const int lane = threadIdx.x % 16;
half16 a_frag;

// Load column 'lane' from A (which is row-major, so we stride by K)
for (int i = 0; i < 16; i++) {
    a_frag[i] = A_lds[i][lane];  // A[row=i][col=lane], col-major access
}
```

---

## B Matrix Fragment Layout

### Memory Layout  
B is conceptually **transposed** for WMMA - each lane loads one "column" which corresponds to one row of the original B matrix:
- Original B[k][j] is at memory offset `k * N + j` (row-major)
- For WMMA, lane L loads B[*][L%16] - all K values for column L%16

### Fragment Register Mapping

For **lane L** (where `effective_lane = L % 16`):

```
b_frag[0]  = B[0][effective_lane]   (packed with b_frag[1])
b_frag[1]  = B[1][effective_lane]
b_frag[2]  = B[2][effective_lane]   (packed with b_frag[3])
b_frag[3]  = B[3][effective_lane]
...
b_frag[14] = B[14][effective_lane]  (packed with b_frag[15])
b_frag[15] = B[15][effective_lane]
```

**Physical Register Layout (8 VGPRs per lane):**

| VGPR | Bits [15:0] | Bits [31:16] |
|------|-------------|--------------|
| v0 | B[0][lane%16] | B[1][lane%16] |
| v1 | B[2][lane%16] | B[3][lane%16] |
| v2 | B[4][lane%16] | B[5][lane%16] |
| v3 | B[6][lane%16] | B[7][lane%16] |
| v4 | B[8][lane%16] | B[9][lane%16] |
| v5 | B[10][lane%16] | B[11][lane%16] |
| v6 | B[12][lane%16] | B[13][lane%16] |
| v7 | B[14][lane%16] | B[15][lane%16] |

### Load Code Pattern (B transposed in LDS)

```cpp
// B_src is [K][N] row-major in global memory
// B_lds is [N][K] in LDS (transposed during load)

const int lane = threadIdx.x % 16;
half16 b_frag;

// Load "row" lane from transposed B
for (int k = 0; k < 16; k++) {
    b_frag[k] = B_lds[lane][k];  // B_lds[n=lane][k]
}
```

---

## C/D Matrix Fragment Layout (FP32 Accumulator)

### Memory Layout
D is stored in **row-major** format:
- `D[i][j]` is at memory offset `i * N + j`

### Fragment Register Mapping

**Key insight: The 32 lanes cover the full 16x16 matrix by having lanes 0-15 cover even rows and lanes 16-31 cover odd rows.**

For **lane L**:
- `col = L % 16` (the column this lane writes to)
- `row_offset = L / 16` (0 for lanes 0-15, 1 for lanes 16-31)

```
c_frag[0] = D[0 + row_offset][col]   // Row 0 (lane 0-15) or Row 1 (lane 16-31)
c_frag[1] = D[2 + row_offset][col]   // Row 2 or Row 3
c_frag[2] = D[4 + row_offset][col]   // Row 4 or Row 5
c_frag[3] = D[6 + row_offset][col]   // Row 6 or Row 7
c_frag[4] = D[8 + row_offset][col]   // Row 8 or Row 9
c_frag[5] = D[10 + row_offset][col]  // Row 10 or Row 11
c_frag[6] = D[12 + row_offset][col]  // Row 12 or Row 13
c_frag[7] = D[14 + row_offset][col]  // Row 14 or Row 15
```

**General formula:**
```
c_frag[i] = D[i*2 + (lane/16)][lane % 16]
```

**Physical Register Layout (8 VGPRs per lane, FP32):**

| VGPR | Lane 0 | Lane 1 | ... | Lane 15 | Lane 16 | Lane 17 | ... | Lane 31 |
|------|--------|--------|-----|---------|---------|---------|-----|---------|
| v0 | D[0][0] | D[0][1] | ... | D[0][15] | D[1][0] | D[1][1] | ... | D[1][15] |
| v1 | D[2][0] | D[2][1] | ... | D[2][15] | D[3][0] | D[3][1] | ... | D[3][15] |
| v2 | D[4][0] | D[4][1] | ... | D[4][15] | D[5][0] | D[5][1] | ... | D[5][15] |
| v3 | D[6][0] | D[6][1] | ... | D[6][15] | D[7][0] | D[7][1] | ... | D[7][15] |
| v4 | D[8][0] | D[8][1] | ... | D[8][15] | D[9][0] | D[9][1] | ... | D[9][15] |
| v5 | D[10][0] | D[10][1] | ... | D[10][15] | D[11][0] | D[11][1] | ... | D[11][15] |
| v6 | D[12][0] | D[12][1] | ... | D[12][15] | D[13][0] | D[13][1] | ... | D[13][15] |
| v7 | D[14][0] | D[14][1] | ... | D[14][15] | D[15][0] | D[15][1] | ... | D[15][15] |

### Store Code Pattern

```cpp
const int lane = threadIdx.x;
const int col = lane % 16;

for (int i = 0; i < 8; i++) {
    int row = i * 2 + (lane / 16);
    D[row * N + col] = c_frag[i];
}
```

---

## Correct Fragment Load from LDS

### For A (stored row-major [M][K] in LDS):

```cpp
__device__ void load_A_fragment(
    half16& a_frag,
    const __half* A_lds,  // [BLOCK_M][BLOCK_K] row-major
    int row_offset,       // Starting row in tile
    int stride            // LDS stride (BLOCK_K)
) {
    const int lane = threadIdx.x % 16;
    
    // Each lane loads its own ROW of the 16x16 A tile
    // Lane 'lane' loads A[row_offset + lane][*] - row 'lane'
    const __half* row_ptr = A_lds + (row_offset + lane) * stride;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        a_frag[i] = row_ptr[i];
    }
}
```

**Note**: When loading from row-major source, each lane loads one **row** of A.
The hardware internally reinterprets this as columns for the matrix multiply.

### For B (stored transposed [N][K] in LDS after transpose from [K][N]):

```cpp
__device__ void load_B_fragment_transposed(
    half16& b_frag,
    const __half* B_lds,  // [BLOCK_N][BLOCK_K] - transposed layout
    int col_offset,       // Starting column in tile
    int stride            // LDS stride (BLOCK_K)
) {
    const int lane = threadIdx.x % 16;
    
    // Each lane loads one row of transposed B (= one column of original B)
    // Lane 'lane' loads B_lds[col_offset + lane][*]
    const __half* col_ptr = B_lds + (col_offset + lane) * stride;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        b_frag[k] = col_ptr[k];
    }
}
```

---

## Using AMD Matrix Instruction Calculator

You can verify these layouts using AMD's official tool:

```bash
# Clone the tool
git clone https://github.com/ROCm/amd_matrix_instruction_calculator.git
cd amd_matrix_instruction_calculator

# Query A matrix layout
./matrix_calculator.py --architecture rdna3 --instruction v_wmma_f32_16x16x16_f16 \
    --register-layout --A-matrix

# Query D matrix layout  
./matrix_calculator.py --architecture rdna3 --instruction v_wmma_f32_16x16x16_f16 \
    --register-layout --D-matrix

# Get detailed instruction info
./matrix_calculator.py --architecture rdna3 --instruction v_wmma_f32_16x16x16_f16 \
    --detail-instruction
```

Note: gfx1151 uses the same WMMA layout as gfx1100 (RDNA3). Specify `--architecture rdna3` or `--architecture gfx1151`.

---

## References

1. [AMD GPUOpen: WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/)
2. [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
3. [RDNA3 ISA Reference Guide](https://developer.amd.com/wp-content/resources/RDNA3_Shader_ISA_December2022.pdf)
4. [rocWMMA Library](https://github.com/ROCm/rocWMMA)

---

## Technical Refinements and Advanced Optimization Lessons

### 1. Vector-Granularity Swizzling for RDNA/LPDDR

On RDNA architecture, banks are fundamentally 32 banks of 4-byte words. For 16-byte vector transactions (`half8`), a half-granularity swizzle can still cause aliasing in "dword space".

**Recommendation**: Define swizzles on the **vector index** (16B chunk index), not on half columns.
- Treat K as `Kvec = K / 8` "half8 columns".
- Store A as `[row][kvec]` with swizzle: `kvec_swz = kvec ^ (row & 1)`.
- This ensures fragment loads read valid numbers from the correct places, avoiding the "high TFLOPS but garbage data" failure mode.

### 2. LDS Fragment Load Optimization

Consider converting LDS fragment loads from 16 scalar half loads into two `ds_read_b128` + pack. Even if the total bytes are identical, this approach:
- Reduces **LGKM overhead**.
- Decreases bank conflict probability.
- Potentially improves performance without pushing VGPR usage over an occupancy cliff.

### 3. LLVM Memory Model and ASYNC LDS

A critical clarification for the AMDGPU memory model:
- **ASYNC LDS and tensor ops are NOT covered** by the memory model implemented by the AMDGPU backend.
- Waits (e.g., `s_waitcnt`) are **not inserted automatically**; they must be emitted explicitly.
- This distinguishes non-ASYNC LDS from ASYNC LDS behavior and informs how we must handle synchronization for advanced tensor operations.

### 4. Deterministic Debugging Primitives

When debugging XOR swizzles or fragment mapping, isolate the following primitives as micro-kernels before integrating into the full GEMM:

- **Roundtrip**: GMEM → LDS (swizzled) → GMEM (unswizzled).
- **Operand Dump**: LDS → WMMA operand fragment → Global memory.
- **Epilogue Dump**: WMMA Accumulator → Global memory using a known test pattern.

Using deterministic patterns (`A[row,col] = row*256 + col`) during these tests ensures that bugs in swizzle math, lane replication, or epilogue mapping are identified in minutes rather than hours.

---

## XOR Swizzle vs Padding: Performance Analysis (December 2025)

### Summary

After implementing and benchmarking XOR swizzle for LDS bank conflict elimination, we found that **padding outperforms XOR swizzle by 15-20%** on gfx1151.

### Benchmark Results

| Approach | TFLOPS | LDS Usage | Ratio |
|----------|--------|-----------|-------|
| **Padding (stride=24)** | 20-21 | 18.4 KB | 1.00x |
| **XOR Swizzle (stride=16)** | 17-18 | 12.3 KB | 0.85x |

### Root Cause Analysis

1. **B Matrix Transpose Stores**: The swizzled kernel requires per-element swizzle computation during B matrix transpose stores:
   ```cpp
   // Swizzled: 8 scalar stores with index computation per thread
   for (int i = 0; i < 8; i++) {
       int phys_idx = Swizzle::to_physical(n_idx, b_k, B_STRIDE);  // Division, modulo, XOR
       B_lds[phys_idx] = data[i];
   }
   
   // Padded: 8 scalar stores with simple 2D indexing
   for (int i = 0; i < 8; i++) {
       B_lds[b_n + i][b_k] = data[i];  // Compiler optimizes 2D array access
   }
   ```

2. **Flat 1D Array Indexing**: The swizzled kernel uses flat 1D arrays with computed indices, while the padded kernel uses 2D arrays that the compiler can optimize better.

3. **RDNA3 LDS Bank Conflict Penalty**: The bank conflict penalty on RDNA3 may not be severe enough to justify the swizzle computation overhead. The padding approach (stride 24 vs 16) effectively breaks bank alignment with minimal overhead.

4. **Fragment Loading Overhead**: Even with optimized vectorized loads, the swizzle requires conditional logic:
   ```cpp
   // Swizzled: Conditional offset based on row parity
   const int grp0_off = (row & 1) ? 8 : 0;
   const int grp1_off = (row & 1) ? 0 : 8;
   
   // Padded: Direct vectorized load
   const half8 v0 = *reinterpret_cast<const half8*>(row_ptr);
   const half8 v1 = *reinterpret_cast<const half8*>(row_ptr + 8);
   ```

### Correctness Fixes Applied

Both `matmul_asmOpt` and `matmul_swizzled` had incorrect fragment loading patterns that were fixed:

**Bug**: Loading COLUMN `frag_col` (iterating rows, fixed column)
**Fix**: Loading ROW `frag_idx` (fixed row, all K values)

This matches the AMD GPUOpen pattern where each lane loads its own row of A (all 16 K values).

### Recommendations

1. **Use padding for gfx1151**: The 33% LDS savings from XOR swizzle doesn't compensate for the ~15-20% performance loss.

2. **XOR swizzle may be beneficial when**:
   - LDS is the limiting resource (need to fit more data)
   - Bank conflict penalty is higher (different architectures)
   - Swizzle computation can be amortized (larger BLOCK_K)

3. **Future optimization**: Consider vector-granularity swizzling (swizzle on half8 chunks, not individual elements) to reduce computation overhead while maintaining bank conflict avoidance.

### All 12 Kernels Now Pass

After fixing the fragment loading patterns, all 12 kernel variants pass correctness tests:

| Kernel | TFLOPS | Status |
|--------|--------|--------|
| matmul_zerocopy | 20.61 | ✅ Best |
| matmul_adaptive | 20.52 | ✅ |
| matmul_asmOpt | 20.49 | ✅ Fixed |
| matmul | 20.46 | ✅ |
| matmul_native | 19.92 | ✅ |
| matmul_kunroll | 18.31 | ✅ |
| matmul_swizzled | 17.90 | ✅ Fixed |
| matmul_noPrefetch | 17.60 | ✅ |
| matmul_xor_optimized | 17.26 | ✅ |
| matmul_quad | 17.01 | ✅ |
| matmul_hilbert | 13.33 | ✅ |
| matmul_highOcc | 10.28 | ✅ |
| matmul_coop | ~20.0 | ✅ New |

---

## Optimization Attempts Summary (December 2025)

This section documents the results of implementing optimizations from the development notes task list.

### Attempted Optimizations

| Optimization | Status | Performance | Notes |
|-------------|--------|-------------|-------|
| **Vectorized C Writes** | ❌ Failed | 0.72x slower | Extra 32KB LDS + sync overhead |
| **Shared Memory Write Buffer** | ❌ Failed | Broken | Union LDS caused correctness issues |
| **Cooperative Loading** | ✅ Implemented | 1.01-1.06x | Marginal, inconsistent gains |
| **Register Prefetching (k+2)** | ✅ Already done | N/A | k+1 reg + k+2 L2 prefetch exists |
| **BLOCK_K=32 (K-unroll)** | ❌ Slower | 0.89-0.97x | LDS/register pressure hurt occupancy |
| **Code Organization** | ✅ Already done | N/A | Headers already extracted |

### Detailed Analysis

#### 1. Vectorized C Writes (FAILED)

**Approach**: Write C fragments to LDS first, then use all threads for coalesced float4 writes to global memory.

**Results**:
- 14.43 TFLOPS vs 20.09 TFLOPS standard (0.72x)
- Extra 32KB LDS for C buffer reduced occupancy
- Additional `__syncthreads()` added latency
- The standard kernel's scalar C writes are already efficient

**Conclusion**: The overhead of staging through LDS outweighs any coalescing benefits.

#### 2. Cooperative Loading (MARGINAL)

**Approach**: Split 256 threads into two halves - 128 load A, 128 load B simultaneously.

**Results by matrix size**:
| Size | Coop | Standard | Ratio |
|------|------|----------|-------|
| 2048×2048×512 | 21.38 TF | 20.19 TF | **1.06x** |
| 4096×4096×512 | 21.09 TF | 20.70 TF | 1.02x |
| 4096×4096×1024 | 21.45 TF | 21.09 TF | 1.02x |
| 4096×4096×2048 | 19.84 TF | 20.48 TF | **0.97x** |
| 8192×8192×2048 | 23.17 TF | 22.61 TF | 1.02x |

**Conclusion**: Helps slightly at small K (1.06x), but hurts at large K (0.97x). Not worth incorporating into the main kernel due to inconsistent gains.

#### 3. BLOCK_K=32 / K-Unroll (SLOWER)

**Approach**: Process 2 WMMA K-tiles (32 elements) per sync to reduce `__syncthreads` overhead.

**Results**:
| Size | K-Unroll | Standard | Ratio |
|------|----------|----------|-------|
| 2048×2048×512 | 20.95 TF | 20.84 TF | 1.01x |
| 4096×4096×512 | 18.69 TF | 20.66 TF | **0.90x** |
| 4096×4096×1024 | 18.99 TF | 21.33 TF | **0.89x** |
| 4096×4096×2048 | 18.78 TF | 19.44 TF | 0.97x |

**Conclusion**: Increased LDS stride (40 vs 24) and register pressure hurt occupancy more than sync savings help.

### Why the Standard Kernel is Hard to Beat

The current standard kernel (`matmul`) is already well-optimized:

1. **Double-buffered LDS** with interleaved prefetch (k+1 to registers)
2. **Software prefetch** for k+2 tiles to L2 cache
3. **Vectorized half8 loads** from global memory
4. **LDS padding** (stride 24) for bank conflict reduction
5. **2×2 register blocking** per warp (4 WMMA tiles)
6. **Interleaved MMA + prefetch** for latency hiding

### Gap to rocBLAS

| Implementation | TFLOPS | % of Peak |
|---------------|--------|-----------|
| **Our best kernel** | 20-21 | 34% |
| **PyTorch/rocBLAS** | 34-37 | 58% |
| **Peak (gfx1151)** | 59.4 | 100% |

The ~43% gap to rocBLAS is likely due to:
- **Split-K parallelism**: rocBLAS can split K across multiple CTAs
- **Persistent kernels**: Avoid kernel launch overhead
- **Assembly-level tuning**: Hand-optimized instruction scheduling
- **Larger tile sizes**: May use 256×128 or larger with careful occupancy tuning

### Recommendations for Future Work

1. **Split-K implementation**: For large K, split across CTAs and reduce
2. **Persistent kernel**: Single kernel launch, tiles fetched from work queue
3. **Assembly optimization**: Use `s_setprio` and manual instruction scheduling
4. **Profile-guided tuning**: Use rocprof to identify actual bottlenecks


# Findings from HipKittens Research (November 2025)

## Overview
Recent research from HazyResearch (HipKittens) highlights several key optimizations for AMD GPUs. We evaluated which techniques are applicable to RDNA3.5 (gfx1151).

## 1. Register Scheduling
- **Finding:** On RDNA3.5, the WMMA instruction uses VGPRs only (no AGPRs). The AGPR optimization from HipKittens does **not apply** to our target.
- **Verified:** ISA inspection shows 0 `v_accvgpr_read/write` instructions.

## 2. Memory Access Patterns & Bank Conflicts
- **Variable Phases:** Unlike NVIDIA's sequential phase assignment, AMD's shared memory access phases vary by instruction (e.g., `ds_read_b128` uses 4 phases/64 banks, while `ds_write_b64` uses 4 phases/32 banks).
- **Swizzling Complexity:** A single swizzle pattern is insufficient for all layouts. For example, writing a 16x16 bf16 tile (`ds_write_b64`) requires a different swizzle pattern than reading a 16x32 bf16 tile (`ds_read_b128`) to avoid bank conflicts. HipKittens solves this with instruction-specific swizzle patterns.

## 3. Scheduling Patterns

### Ping-Pong Scheduling
- **Tested:** Implemented 4-wave ping-pong pattern (adapted from 8-wave for RDNA3.5's 2 SIMDs/CU)
- **Result:** Slower than baseline (10.4 vs 16 TFLOPS)
- **Reason:** For GEMM, all waves must accumulate all K-tiles—no wave specialization possible

## 4. L2-Aware Tile Mapping
- **Implemented:** `l2_aware_tile_mapping()` in `wmma_tile_mapping.hpp`
- **Benefit:** Groups tiles into 2D super-tiles for better cache locality
- **Note:** Chiplet swizzling is available for future multi-XCD GPUs

## Summary for RDNA3.5 (gfx1151)

| HipKittens Technique | Applicable? | Status |
|---------------------|-------------|--------|
| Register scheduling (AGPRs) | ❌ No | RDNA uses VGPRs only |
| Bank conflict swizzling | ✅ Yes | Implemented (XOR swizzle) |
| Ping-pong scheduling | ❌ No | Tested, slower for GEMM |
| L2-aware tile mapping | ✅ Yes | Implemented |

## XOR Swizzle Review Results (December 2024)

### ISA Analysis
Examined generated assembly for LDS instructions:

**Before optimization:**
| Instruction | Count | Notes |
|------------|-------|-------|
| `ds_load_b128` | 226 | ✅ Optimal 128-bit reads |
| `ds_store_b16` | 216 | ❌ Scalar stores |
| `ds_store_b16_d16_hi` | 212 | ❌ Scalar stores |
| `ds_store_b128` | 60 | ✅ Vectorized stores |

**After optimization (vectorized B matrix stores):**
| Instruction | Count | Change |
|------------|-------|--------|
| `ds_load_b128` | 226 | — |
| `ds_store_b16` | 208 | -8 |
| `ds_store_b16_d16_hi` | 204 | -8 |
| `ds_store_b128` | 64 | +4 |

### Root Cause
The B matrix transpose scatter pattern (loading 8 contiguous N-values, scattering to 8 rows) forced scalar stores. Changed to gather pattern (loading 16 K-values per row) to enable vectorized stores.

### Remaining Limitation
The B matrix gather from global memory uses strided loads (stride = N), which cannot be vectorized. Further improvement requires:
1. Pre-transposing B on the host
2. Using a different B global layout
3. Using async copy with transpose hardware (if available)

### Correctness
Verified with 256×256×256 GEMM: max error = 0.023 (within FP16 precision).

## Chiplet Swizzling Implementation (December 2024)

Added L2/LLC-aware chiplet swizzling to `wmma_tile_mapping.hpp` based on HipKittens Algorithm 1.

### New Functions

1. **`chiplet_swizzle_tile_mapping<BLOCK_M, BLOCK_N>()`**
   - For multi-XCD GPUs (MI300X, MI350X, MI355X)
   - Two-step algorithm:
     - XCD Grouping: Chunks of consecutive block IDs go to same XCD
     - Windowed Traversal: Vertical windows for L2 reuse

2. **`l2_aware_tile_mapping<L2_TILE_M, L2_TILE_N, BLOCK_M, BLOCK_N>()`**
   - For single-XCD GPUs (RDNA3/3.5 like gfx1151)
   - Groups tiles into 2D super-tiles for L2 locality

3. **`get_chiplet_params(gpu_arch, ...)`**
   - Returns recommended parameters per GPU architecture
   - gfx1151: 1 XCD, window_height=4, chunk_size=16
   - gfx942 (MI300X): 8 XCDs, window_height=4, chunk_size=32
   - gfx950 (MI355X): 8 XCDs, window_height=8, chunk_size=32

### New Tile Mapping Modes

- `TileMappingMode::CHIPLET_SWIZZLE` - Full XCD-aware swizzling
- `TileMappingMode::L2_AWARE` - 2D super-tile grouping

### Selection Logic

Updated `select_tile_mapping()`:
- Small matrices (<64 tiles): ROW_MAJOR
- Large matrices (>=256 tiles): L2_AWARE
- Square matrices: HILBERT
- Rectangular: SWIZZLE

## Explicit Register Allocation Investigation (December 2024)

### ISA Analysis for gfx1151 (RDNA3.5)

**Key Finding: No AGPR Issues on RDNA3**

The HipKittens paper describes AGPR/VGPR register movement issues on CDNA GPUs (MI300X), but this does **NOT apply to gfx1151**:

| Aspect | CDNA (MI300X/gfx942) | RDNA3.5 (gfx1151) |
|--------|---------------------|-------------------|
| Matrix instruction | `v_mfma_*` | `v_wmma_*` |
| Register types | AGPRs + VGPRs | VGPRs only |
| Register split | 256 AGPRs + 256 VGPRs | 256 VGPRs total |
| AGPR issue | HIPCC can't use AGPRs as MFMA inputs | N/A - no AGPRs |

### ISA Inspection Results

```
Instruction counts:
- v_wmma_f32_16x16x16_f16: 114 instances
- v_accvgpr_read/write: 0 instances (not used on RDNA3)
- v_mov_b32: 1340 instances (normal register shuffling)
```

**Register usage per kernel:**
- VGPR: 57-92 per kernel
- SGPR: 20-28 per kernel
- No AGPRs (RDNA3 doesn't use them)

### Conclusion

Explicit register allocation optimizations from HipKittens are **NOT applicable** to gfx1151 because:
1. RDNA3 uses WMMA, not MFMA
2. No AGPR/VGPR split exists
3. No `v_accvgpr_read/write` inefficiency to avoid

## 8-Wave Ping-Pong Kernel Implementation (December 2024)

### Overview

Implemented the HipKittens ping-pong scheduling pattern in `wmma_kernels_pingpong.hpp`.

**Adaptation for RDNA3.5 (gfx1151):**
- RDNA3.5 has 2 SIMDs per CU
- We use **4 logical wave pairs** (8 warps of 32 threads each)
- Even wave pairs (0,2) start as "compute"
- Odd wave pairs (1,3) start as "memory"
- Roles swap at each cluster boundary

### Key Features

1. **Wave-pair role alternation**:
   ```cpp
   bool is_compute_role = (wave_pair_id & 1) == 0;
   // Toggle at cluster boundaries
   is_compute_role = !is_compute_role;
   ```

2. **AMD scheduling hints**:
   ```cpp
   __builtin_amdgcn_s_setprio(1);     // Raise priority for compute
   __builtin_amdgcn_sched_barrier(0); // Flush scheduler state
   __builtin_amdgcn_s_barrier();      // Thread barrier
   ```

3. **Cluster structure per K-tile**:
   - Cluster 0: Compute waves do WMMA, memory waves prefetch
   - Barrier
   - Cluster 1: Roles swapped
   - Barrier

### ISA Verification

Generated assembly shows scheduling hints are being emitted:
- `s_setprio 1/0` for priority raising/lowering
- `sched_barrier mask(0x00000000)` for scheduler flush
- `s_barrier` for synchronization

### Files Added

- `wmma_kernels_pingpong.hpp` - New ping-pong kernel implementation
- Added include to `wmma_gemm.hip`

### Expected Benefits

- Better compute/memory overlap within a CU
- Higher effective SIMD utilization
- Reduced pipeline stalls

### Next Steps

1. Add Python binding for `matmul_pingpong()` function
2. Performance benchmarking vs baseline
3. Tune cluster granularity if needed

## Ping-Pong Kernel Results (December 2024)

### Implementation

Created `wmma_kernels_pingpong.hpp` with HipKittens-inspired scheduling.

### Results

| Kernel | Correctness | Performance (1024³) |
|--------|-------------|---------------------|
| Ping-pong | ✅ PASS (error 0.016) | 10.42 TFLOPS |
| Adaptive (baseline) | ✅ PASS | 16.03 TFLOPS |

### Analysis: Why Ping-Pong Doesn't Help GEMM on RDNA3.5

1. **Architecture**: RDNA3.5 has 2 SIMDs per CU (HipKittens targets 4 SIMDs)

2. **GEMM vs Attention**: HipKittens ping-pong is designed for attention where:
   - Different operations (QK, softmax, AV) can be overlapped
   - Waves can specialize in different roles

   For GEMM, **all waves must accumulate all K-tiles** - no specialization possible.

3. **Already optimal**: The standard double-buffering in `matmul_adaptive` already achieves good compute/memory overlap.

### Conclusion

The ping-pong pattern is **not beneficial for GEMM on RDNA3.5**. The kernel is correct but slower than the baseline.

**Recommended approach for RDNA3.5 GEMM optimization:**
- Focus on tile sizes and occupancy tuning
- Use chiplet swizzling (already implemented in `wmma_tile_mapping.hpp`)
- Optimize LDS bank conflicts (XOR swizzle)

## Session Cleanup (December 2024)

### Removed Files
- `flash_attention.hpp` - Removed attention kernel (focus is GEMM)
- `examples/hipkittens_gqa_kernel.cpp` - Removed (re-added by user for reference)
- `WMMA_DEVELOPMENT_NOTES.md.orig` - Cleanup backup file

### Current Focus: Maximum WMMA GEMM TFLOPS

**Target**: Achieve peak TFLOPS on gfx1151 (59.4 TFLOPS theoretical)

**Current Best**: `matmul_adaptive` at ~16 TFLOPS (27% efficiency)

### Available GEMM Kernels

| Kernel | Description |
|--------|-------------|
| `matmul` | Base 128×64 tile, double-buffered |
| `matmul_adaptive` | Auto tile selection (best performance) |
| `matmul_hilbert` | Hilbert curve tile mapping |
| `matmul_pingpong` | HipKittens-style scheduling (slower) |
| `matmul_kunroll` | K-unrolling (2x fewer syncs) |
| `matmul_xor_optimized` | XOR-swizzled LDS |

### Next Steps for Higher TFLOPS
1. Profile with `rocprof` to identify bottlenecks
2. Tune tile sizes for different matrix shapes
3. Investigate memory bandwidth utilization
4. Consider larger tiles (256×128, 256×256)

# Analysis of adelj88/rocm_wmma_gemm Optimizations (December 2024)

## Overview

Reviewed high-performance WMMA GEMM kernels from [adelj88/rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm).

**Key achievements (RX 7900 GRE):**
- 76.37 TFLOPS (8192³), 94.2% of rocBLAS
- Up to 138% of rocBLAS on some LLM shapes

## Key Optimizations to Incorporate

### 1. Larger Tile Sizes (256×256)

| Config | Our Current | adelj opt_5 |
|--------|-------------|-------------|
| Block M | 128 | 256 |
| Block N | 64 | 128 |
| Warp Tile M | 2 | 4 |
| Warp Tile N | 2 | 4 |
| Accumulators/warp | 4 | 16 |

**Action:** Implement 256×128 tile size with 4×4 warp tiling.

### 2. Cooperative Loading (Half-Block Split)

Adelj uses **half-threads load A, half-threads load B** pattern:
```cpp
if(tid < half_block) {
    // Load A tile
} else {
    // Load B tile
}
```

**Benefit:** Better memory bandwidth utilization - both matrices streamed simultaneously.

### 3. FP16 Accumulator + FP16 Output

Adelj uses `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32()` for FP16→FP16 output.

**Our current:** `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32()` (FP16→FP32)

**Trade-off:** FP16 accum = faster, less precision. Consider for inference.

### 4. Vectorized Epilogue via LDS

Two-phase epilogue:
1. Store fragments to LDS
2. Vectorized coalesced writes from LDS to GMEM

```cpp
// Phase 1: Store fragments to LDS
c_tile[row * block_n + col] = c_frags[wm][wn][i * 2];
__syncthreads();

// Phase 2: Vectorized GMEM write
*reinterpret_cast<vector_type*>(C + offset) = 
    *reinterpret_cast<vector_type*>(c_tile + offset);
```

**Benefit:** Coalesced 256-bit writes vs scattered 32-bit stores.

### 5. Optimized Fragment Loading

Different loading pattern - loads fragments row-by-row in a loop:
```cpp
for(int i = 0; i < wmma_tile; ++i) {
    const half* srca = curr_a + (i * lds_stride_A);
    for(int wm = 0; wm < warp_tile_m; ++wm) {
        a_frag[wm][i] = *srca;
        srca += wmma_tile;
    }
}
```

### 6. Hilbert Curve with Remainder Handling

Improved Hilbert mapping that handles non-power-of-2 grids:
- Core power-of-2 region: Hilbert curve
- Remainder regions: Row-major fallback

Already have similar in `wmma_tile_mapping.hpp` but their version is cleaner.

### 7. `__launch_bounds__` for Register Control

```cpp
__launch_bounds__(warp_size * config::total_warps) kernel_hgemm(...)
```

Explicit thread count helps compiler optimize register allocation.

## Implementation Priority

1. **HIGH:** Vectorized LDS epilogue (fix our scattered stores)
2. **HIGH:** Larger tiles (256×128 or 256×256)
3. **MEDIUM:** Cooperative loading (half A, half B)
4. **LOW:** FP16 accumulator option for inference

## Performance Gap Analysis

| Metric | Our Best | adelj opt_5 | Gap |
|--------|----------|-------------|-----|
| TFLOPS (1024³) | 16 | ~45 (estimated) | 2.8x |
| % of peak | 27% | ~75% | - |

**Root causes of gap:**
1. Smaller tiles (fewer MACs per LDS load)
2. Scattered epilogue stores
3. Less efficient fragment loading

---

## Interleaved MMA + Prefetch Optimization (December 2025)

### Summary

Implemented and verified the **interleaved MMA + prefetch pattern** across all standard GEMM kernels. This optimization hides memory latency by overlapping global memory loads with WMMA compute operations.

### The Pattern

The key insight is that WMMA operations have significant latency, and we can use this time to prefetch data for the next iteration:

```cpp
// Sequential (SLOW - 19.4 TFLOPS):
prefetch_A();
prefetch_B();
mma_sync(c00, a0, b0, c00);
mma_sync(c01, a0, b1, c01);
mma_sync(c10, a1, b0, c10);
mma_sync(c11, a1, b1, c11);
write_to_lds();

// Interleaved (FAST - 20.6 TFLOPS):
mma_sync(c00, a0, b0, c00);  // MMA while prefetching A
a_prefetch = load(A_base + BLOCK_K);

mma_sync(c01, a0, b1, c01);  // MMA while prefetching B
b_prefetch = load(B_base + BLOCK_K * stride);

mma_sync(c10, a1, b0, c10);  // MMA while software prefetch k+2
__builtin_prefetch(A_base + 2 * BLOCK_K);
__builtin_prefetch(B_base + 2 * BLOCK_K * stride);

mma_sync(c11, a1, b1, c11);  // MMA while writing to LDS
write_to_lds(a_prefetch, b_prefetch);
```

### Performance Impact

| Pattern | TFLOPS | % of Peak |
|---------|--------|-----------|
| Sequential (prefetch before MMA) | 19.4 | 32.7% |
| **Interleaved (prefetch during MMA)** | **20.6** | **34.7%** |

**~6% improvement** from interleaving alone.

### Kernels Updated

| Kernel | Status | Notes |
|--------|--------|-------|
| `wmma_gemm_kernel` | ✅ Fixed | Main kernel |
| `wmma_gemm_kernel_gfx1151` | ✅ Fixed | Used by `matmul_adaptive` |
| `wmma_gemm_kernel_alphabeta` | ✅ Fixed | Alpha/beta scaling |
| `wmma_gemm_kernel_asmOpt` | ✅ Already correct | Had pattern from start |

### Kernels with Different Structures

These kernels use different optimization strategies and don't benefit from the same interleaving:

| Kernel | Structure | Reason |
|--------|-----------|--------|
| `wmma_gemm_kernel_kunroll` | K-unrolling | Processes 2 K-tiles per sync |
| `wmma_gemm_kernel_noPrefetch` | No prefetch | Baseline without prefetch |
| `wmma_gemm_kernel_quad` | Quad-buffering | 4-buffer approach |
| `wmma_gemm_kernel_hilbert` | Hilbert mapping | Pointer swapping structure |
| `wmma_gemm_kernel_coop` | Cooperative loading | Split A/B thread groups |

### Key Implementation Details

1. **Register-staged prefetch**: Use `half8` variables to hold prefetched data while MMA executes
2. **Pointer increment optimization**: Pre-compute base pointers and increment at loop end
3. **Software prefetch for k+2**: Use `__builtin_prefetch` to bring data into L2 cache
4. **Bounds checking**: Check `k + BLOCK_K + offset < K` for next iteration

### File Reorganization

Renamed header files for clarity:
- `wmma_kernels_opt.hpp` → `wmma_kernel_largetile.hpp` (single 128×128 kernel)
- `wmma_kernels_optimized.hpp` → `wmma_kernel_variants.hpp` (multiple variant kernels)

---

## Current Performance Summary (December 2025)

| Kernel | TFLOPS | % Peak | Status |
|--------|--------|--------|--------|
| matmul | 20.5 | 34.5% | ✅ Best |
| matmul_adaptive | 20.4 | 34.3% | ✅ |
| matmul_asmOpt | 19.5 | 32.8% | ✅ |
| matmul_native | 19.4 | 32.7% | ✅ |
| matmul_zerocopy | 19.2 | 32.3% | ✅ |
| matmul_kunroll | 18.9 | 31.8% | ✅ |
| matmul_noPrefetch | 18.0 | 30.3% | ✅ |
| matmul_quad | 16.9 | 28.5% | ✅ |
| matmul_hilbert | 14.1 | 23.7% | ✅ |
| matmul_highOcc | 10.3 | 17.3% | ✅ |

**PyTorch/rocBLAS reference**: 37.4 TFLOPS (63% of peak)

All 12 kernel variants pass correctness tests with ~0.026% relative error.

---

## 2026-08-23 frontier campaign update

This section records the current campaign without rewriting the historical
entries above.  The full experiment ledger, including rejected variants and
reproduction details, is in
[`FRONTIER_EXPLORATION_2026-08-23.md`](../FRONTIER_EXPLORATION_2026-08-23.md).
The public repository record and the persistent-prepacked experiments use
different numerical and input-layout contracts and must not be conflated.

### Current reference points

| Contract | Result | Status |
|---|---:|---|
| Repository FP16 input / FP32 accumulation and output | 41.322 TFLOPS peak | Full rocBLAS-reference validation; five-process median 40.900 TFLOPS, so the sustained promotion gate remains open |
| Upstream FP16 input / FP16 output control | 46.082 TFLOPS median | Reproduced from three fresh processes; external comparison, not the repository record |
| Original-layout FP16-output control in the block-pack campaign | 45.697 TFLOPS | Same-pass screening control |
| Persistent block/K-major FP16-output inputs | **49.035 TFLOPS average** | Five fresh 100-iteration processes; current sustained leader for the separate prepacked-input contract |
| Best short persistent block/K-major pass | 49.573 TFLOPS | Below the 50 TFLOPS gate and not sustained |

The retained persistent-input leader is a 256x128 N-packed, padding-8 kernel.
Its K16 loop contains 16 WMMAs, 16 `ds_load_b128` instructions, three
cooperative `global_load_b128` instructions, three `ds_store_b128`
instructions, nine waitcnts, and two barriers.  It uses 118 VGPR, 22 SGPR, and
18 KiB LDS and reproduces the tightened error tuple: normalized maximum error
0.018779343, RMS error 0.035428338, and cosine similarity 0.999977929.

Two isolated short samples crossed 50 TFLOPS (50.216 for the control schedule
and 50.338 for a progressive LDS schedule), but both regressed in longer
same-pass comparisons.  They are recorded as power/noise excursions, not
records.  Promotion still requires correctness plus a sustained improvement
across fresh processes.

### What the latest experiments established

- Inter-wave producer/consumer kernels remain register-allocation limited on
  gfx1151.  Producer and consumer paths inherit one kernel-wide allocation;
  reducing consumer state restores occupancy only by removing independent
  WMMA chains.  Measured variants reached 5.747--7.098 TFLOPS.
- Explicit in-wave transposes do not beat native scalar LDS fragment loads.
  The best one-sided 2x2 DPP form reached 25.545 TFLOPS.  DPP work and live
  state cost more than the existing LDS multicast/bank behavior saves.
- Whole-matrix native-order prepacking loses global coalescing.  Block/K-major
  16-wide microtiles are the useful contract: they preserve cooperative b128
  loads while presenting operands in WMMA fragment order.
- Padding both A and B by eight halves is jointly necessary.  Removing either
  padding side falls to 44.85--45.71 TFLOPS.  XOR-snake swizzle 16 remains the
  best measured block mapping within the kernel.
- Loop unrolling, compiler-generated ping-pong, 16-wave geometry, compact XOR
  LDS layouts, paired LDS reads, alternate allocation offsets, and scheduler
  flag sweeps did not improve the retained leader.
- Separate rocprofiler-Compute collection attributes 14.6% of wave cycles to
  barrier waits and 11.7% to waitcnt stalls for the block-packed kernel.  A
  one-barrier K step remains the measured goal, provided it preserves the
  leader's two-block, 16-wave residency.

### Hand-scheduled K32 and queued follow-ups

A K32 slice-major two-slot ring now packs operands as
`[block][K32][slice][row][K16]`.  The first C++ form used 202 VGPR and reached
39.292 TFLOPS; the dedicated form used 151 VGPR.  ISA inspection found LLVM
hoisting all four B fragments, extending 24 unnecessary VGPRs.  The current
hand-scheduled assembly reuses one eight-register B fragment and assembles at
128 VGPR with no spills and 30,720 bytes of LDS.  The initial validation
failure was a schedule bug: B fragments were read after refill stores had
started replacing the slot.  Loading B first restored the exact reference
tuple.  Interleaving refill VMEM between B groups reached 41.988 TFLOPS
sustained (43.667 short), versus 47.767 for the same-pass p8 control.  A
two-B-fragment form with recomputed addresses retained 128 VGPR but reached
only 39.921 TFLOPS.  K32 barrier reduction is correct but rejected on speed.

The 5x8 and cyclic-skewed 5x8 workgroup mappings also passed correctness and
reached 46.640 and 45.985 TFLOPS respectively, versus a 47.793 TFLOPS
same-pass control.  Composable Kernel's `s_setprio` pattern reached 42.439
versus 47.937 TFLOPS.  A wait-after-barrier schedule was neutral (47.789
versus 47.889 TFLOPS), while split `s_barrier_signal`/`s_barrier_wait`
instructions are not supported by the gfx1151 assembler.

A distinct wave-private-A architecture gives each wave its own 64x16 A tile
and double-buffers the shared B tile.  It uses exactly 32 KiB LDS, 123 VGPR,
and no spills, retaining two blocks/16 waves per CU while reducing the K16
loop to one barrier.  It passed the full-reference tuple but reached only
37.707 TFLOPS; duplicate A loads for paired N waves erase the synchronization
saving.

### IU4 ISA and hardware qualification

The gfx1151 `v_wmma_i32_16x16x16_iu4` path has now been read from the RDNA3.5
ISA, checked with AMD's Matrix Instruction Calculator, compiled through the
ROCm 7.14 builtin, and measured.  Each lane supplies two packed-nibble VGPRs
for A and B and eight I32 accumulator VGPRs.  `NEG[0]`/`NEG[1]` select signed
or unsigned interpretation.  The instruction performs 8192 integer operations
in 16 cycles; at 20 WGPs and 2.9 GHz its nominal ceiling is 118.8 TOPS.

The standalone harness validates the full nonuniform 16x16 product, including
the repeated A/B operand halves and the even/odd-row D mapping.  It then uses
independent accumulator chains for issue-rate measurement.  All checks pass;
1/2/4/8/12/16 chains reached 103.040/98.107/108.380/109.408/109.718/110.229
INT4 TOPS.  Four chains are already within 1.7% of the best result, an important
register-budget result for a real kernel.

This is not directly usable by the current ROCmFP4 path.  Codebook10 is a
nonlinear signed codebook with magnitudes through 10, not a linear signed-IU4
encoding, and activations are not currently W4.  A direct exact sum-of-linear
weight decomposition needs at least two IU4 products; a single IU4 still needs
nonlinear correction work, before activation decomposition and scaling.
Future work should therefore treat linear W4A4 as a new quantization/quality
contract and retain the existing Codebook10 DP4A path until that contract
passes the model quality gates.

The follow-up full GEMM prototype uses block/K-major linear INT4 inputs and
exact INT32 output.  Its retained 128x128 schedule has eight waves, a 4x2
WMMA-tile footprint per wave, eight independent accumulators, double-buffered
4 KiB LDS, and one barrier per K16.  LLVM allocates 89 VGPR and 22 SGPR with
no spills.  The two-DWORD LDS row is deliberately unpadded: 16 unique lanes
span the 32 banks exactly once and the upper wave half is a replicated
multicast.  Padding to three DWORDs fell to 83.416 TOPS.

The complete 4096-cubed kernel reached 85.907 INT4 TOPS (1.599854 ms) and
matched all 16,777,216 outputs exactly.  Same-pass geometry brackets favored
four wave columns at 85.063/85.907 TOPS over two columns at 84.992/84.182.
A 256x128 tile used 159 VGPR and reached only 78.367 TOPS.  This is the first
new architecture in the campaign to exceed 50 under its own full GEMM
contract, but the unit is integer and the inputs are prequantized: it is not a
50-TFLOPS FP16 result.  The next inference-facing gate is model-quality testing
of a linear W4A4 quantizer and the cost of activation packing/scales.

Current literature supports the same constraint observed experimentally:
Tawa, HipKittens, and FIBER all highlight that effective producer/consumer
specialization depends on asynchronous copy/barrier facilities or dynamic
register sharing that gfx1151 does not provide.  AMD FlyDSL reinforces the
use of 128-bit cooperative copies and explicit load/WMMA/store schedule groups.
These references informed the queued experiments; they are not performance
evidence by themselves.

### Compact ping-pong and wide-workgroup closure

The final occupancy-preserving barrier pass tested three new FP16-output
architectures rather than retuning the retained loop:

- Direct A with double-buffered shared B was exact at 140 VGPR and 12 KiB LDS,
  but duplicate wave-level A traffic limited it to 26.790 TFLOPS.
- A persistent 80-workgroup kernel kept one 256-row A block resident while
  walking N shards.  Static outer-loop unrolling removed spills, but allocation
  remained 256 VGPR, occupancy fell to one block, and throughput was 35.709
  TFLOPS.
- A compact interleaved ping-pong layout placed buffer 0 in halves 0--15 and
  buffer 1 in halves 24--39 of an LDS row.  The 40-half pitch fits both A/B
  buffers in 30 KiB and needs one publish barrier per K16.  The C++ form used
  186 VGPR and reached 40.757 TFLOPS.  Hand scheduling restored 118 VGPR, zero
  spills, and two blocks/16 waves, but the best placement reached 46.024
  TFLOPS versus roughly 48.1 for its bracketed p8 control.

Pitches 32--42 and both legal end placements were screened.  Only pitch 40
retained useful throughput; neighboring correct forms clustered near 22
TFLOPS.  A second compact construction alternated 24- and 16-half A-row
strides, leaving B at p8 and fitting two complete ping-pong tiles in exactly
32 KiB.  Both parity orientations passed full validation at 44.912--44.947
TFLOPS.  The exercise is a useful warning: balanced static bank counts do not
capture the actual LDS issue schedule.

A 512x128 geometry then doubled work per workgroup and B reuse while retaining
the ordinary 64x64 per-wave tile.  Selective B loaders compiled at 137 VGPR and
37.653 TFLOPS.  Duplicating cache-resident B loads removed the divergent path
and cut allocation to 124 VGPR, but gfx1151 still admitted one 16-wave block
per CU and throughput fell to 35.850 TFLOPS.  Larger workgroups do not create
more latency-hiding waves on this target.

The independent A/B padding sweep confirms p8 as the unique single-buffer
optimum, including previously untested odd strides.  A reversible CPU-EPP
bracket changed the p8 leader by only 0.35% with the GPU fixed at 2.9 GHz.
At this stage the retained result was still 48.614 TFLOPS sustained under the
block/K16-prepacked FP16-output contract; the later register-phase result
supersedes it, while the 50-TFLOPS gate remains unmet.

The follow-on resident-fragment and instruction-policy checks also closed
without a promotion.  Keeping all four B fragments resident while streaming A
used 135 VGPR and remained exact, but reached 44.938 TFLOPS.  A/DLC,
A/GLC, A/SLC, B/DLC, B/GLC, B/SLC, and both/SLC cache-policy variants reached
47.956, 47.386, 47.246, 46.770, 47.973, 47.721, and 47.307 TFLOPS inside a
48.263/47.933 control bracket.  Default caching remains best.

Grouping all three refill loads in one clause reached 48.387 TFLOPS, splitting
them two-plus-one reached 48.254, and removing the clause reached 48.540,
against 48.340 and 48.481 controls.  All were exact; the largest delta was
0.12% over the closing control and is noise, not a new schedule.  The existing
two-B-fragment source option emitted byte-identical device ISA to the control
under the pinned ROCm 7.14 compiler, so it does not define another candidate.

The subsequent prefetch/handoff campaign turned several failures into useful
boundaries.  Direct B with shared A reached 32.774 TFLOPS.  A correct pair-local
A handoff replaced one hardware barrier with LDS flags and scalar polling, but
reached only 44.723 TFLOPS at 142 VGPR and 24,608 bytes LDS; its cyclic-prefetch
form fell to 37.298.  Front-loading all next-tile loads into dedicated VGPRs
reached 45.714 TFLOPS with flat loads and 46.009/46.100 with vector/scalar
MUBUF.  These results show that late register reuse and compact issue placement
are worth more than nominal prefetch distance on this kernel.

The vector epilogue also supplied an ISA correction.  RDNA 3.5 DPP
`bank_mask` selects four-lane groups, not lane-id bits.  The initial transpose
was therefore invalid.  A corrected full-bank DPP plus lane-select network
passed the exact reference tuple at 47.335 TFLOPS versus 48.255/48.341 controls;
the epilogue is not the missing throughput.  Removing the explicit source
VMEM wait was neutral at 48.179 versus 48.325/48.151 controls.

Scalar-offset MUBUF recurrence remains a small research building block, not a
promotion.  A short pass reached 48.692 TFLOPS, while four longer interleaved
runs averaged 47.786 versus 47.613 TFLOPS controls, a reproducible 0.36% gain.
It preserves 118 VGPR, 22 SGPR, 18 KiB LDS, two blocks/16 waves, and zero
spills.  Clause removal, late SALU recurrence, and advancing only B by four or
eight WMMA slots did not improve it.  Failures here distinguish a visible
stall counter from work that can actually be moved profitably.

### Shared-box validation discipline

Every model-loading or GPU profiling command must acquire `/root/gpu.lock`.
Only one large model fits in the shared UMA budget; a second loader can silently
fall back to hybrid expert placement and invalidate both performance and token
comparisons.  Before treating a mismatch as a regression, check free memory and
stale containers.  The supported pattern is:

```bash
flock -w 7200 /root/gpu.lock -c '<exclusive validation command>'
```

Correctness runs before timing, profiler counters are collected in separate
passes, and only a correct same-pass improvement advances to the sustained
fresh-process promotion gate.

### 2026-08-23: Paperclip-guided register live-range experiment

A Paperclip full-text pass over recent GEMM/compiler papers produced one new
implementable experiment rather than another wholesale architecture port.
Nautilus recommends splitting long local-buffer live ranges and
rematerializing inexpensive values near use; VeriLocc shows that register
assignment can hide performance opportunities even in mature AMD toolchains.
The newer FP16 PTX study and GPU-Tile-Sim add the guardrails: occupancy alone
does not predict throughput, and the full dependency/overlap graph must be
measured.

The four-wave 128x128 p8 kernel initially compiled at 153 VGPR. An opt-in
`WMMA_BP_STREAM_B_BARRIER` scheduling fence stops LLVM from hoisting all four B
fragments and lowers it to 129. `WMMA_BP_LATE_B_REFILL` then overlaps only A
with WMMA, commits A after the handoff barrier, and streams B into LDS, lowering
the count to 121. `WMMA_BP_BUFFER_A_PREFETCH` expresses both A vector loads as
MUBUF operations with one shared vector offset and an immediate `+16` on the
second load, reaching 120 VGPR, 22 SGPR, 12 KiB LDS, and zero spills.

This reaches a real 24-VGPR allocation boundary; the earlier 129-to-127 change
did not. The ordinary default device assembly remains byte-identical apart
from its generated HIP CUID.

The gfx1151 bracket then supplied the missing result. The 153-VGPR four-wave
control reached 41.419 TFLOPS at four blocks/16 waves. Streaming B at 129 VGPR
raised residency to five blocks/20 waves and reached 45.694 TFLOPS, a 10.3%
gain. Further lowering did not change residency: 127-VGPR late-B1 reached
45.011, 121-VGPR split refill reached 44.630, and the 120-VGPR MUBUF split
reached 44.839 TFLOPS. All passed the full rocBLAS reference tuple. Bracketed
129-VGPR repeats were 45.585/45.598 versus 48.031/48.048 for the retained p8
kernel.

That first result established a joint-resource hypothesis, not a single-cause
answer: five 12-KiB workgroups consume 60 KiB LDS, while the 129-VGPR streamed
form remains in the 144-register allocation class. We therefore swept both
constraints separately and together.

With the 129-VGPR streamed schedule, p8p2/p4p4/p2p0/p0p0 reduced LDS to
10.5/10/8.5/8 KiB and reached 40.608/26.696/37.495/45.445 TFLOPS. The host
occupancy API reported five blocks/20 waves for every form. Zero padding is a
surprising near-neutral LDS layout for this schedule; the intermediate pitches
are not, and p4p4 is catastrophic.

Adding split MUBUF refill lowered the same forms to 124/122/120/119 VGPR. The
p8p2 and p4p4 variants did report six blocks/24 waves, but reached only
40.854/26.495 TFLOPS. The p2p0 and p0p0 variants still reported five blocks/20
waves and reached 35.426/43.796 TFLOPS. Every candidate passed the full
rocBLAS reference tuple. Bracket controls were 45.827/45.698 TFLOPS for the
129-VGPR p8 stream and 48.001/48.234 for the retained p8 kernel.

Failure supplied the decisive lessons. Six reported blocks do not compensate
for an unfavorable LDS bank phase, and removing B-load overlap costs speed
when residency does not rise. Also, do not infer occupancy from `.vgpr_count`
alone: it drops monotonically to 119, but `.amdhsa_next_free_vgpr` stays 169
and the host occupancy query is non-monotonic. Retain the 129-VGPR p8 streamed
implementation as the best four-wave form, but do not promote it or run a
longer screen.

Paperclip also extracted [TileFuse](https://arxiv.org/abs/2606.11357). Its
XDNA2 implementation is not portable, but its offline layout rule is directly
useful to Ember: store each pre-tiled quantized weight block in physical
consumer order and colocate its scales/metadata, so the runtime kernel need
not gather or materialize them. This is an inference-layout avenue, not an FP16
record claim.

Primary papers:

- [Hand-Written PTX Tensor-Core GEMM Kernels](https://arxiv.org/abs/2608.10103)
- [Nautilus](https://arxiv.org/abs/2604.14825)
- [VeriLocc](https://arxiv.org/abs/2506.17506)
- [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262)
- [TileFuse](https://arxiv.org/abs/2606.11357)

### 2026-08-23: Hybrid A-only and B-only ping-pong

Full p8/p8 ping-pong needs 36 KiB LDS and loses the retained kernel's
two-block residency. Two asymmetric designs were built to preserve residency
and bank phase while moving only the safe refill stores ahead of the overwrite
barrier:

- A-only ping-pong uses 30 KiB LDS. The next A tile is stored to its inactive
  buffer during WMMA; B remains single-buffered and is committed between the
  two existing barriers.
- B-only ping-pong uses 24 KiB LDS. B is issued before both A loads,
  `vmcnt(2)` retires only that oldest operation, and the B store overlaps WMMA;
  A remains in the serial handoff.

The first A-only compiler form used 151 VGPR because LLVM extended fragment
and buffer-address lifetimes. A scheduling fence after each four-WMMA B group
restored 127 VGPR with zero spills. Unrolling the two A-buffer phases as
compile-time constants instead raised allocation to 177 VGPR and was discarded
without a GPU run. The B-only form compiled at 129 VGPR. Both use 22 SGPR.

Every timed form reproduced the full rocBLAS tuple and the host API reported
two blocks/16 waves:

| Form | TFLOPS | Same-pass p8 controls |
|---|---:|---:|
| A-only, split loads | 44.921 / 44.619 | 48.082 / 48.079 |
| A-only, grouped loads | 44.606 / 44.569 | 48.082 / 48.079 |
| A-only, late `vmcnt(1)` | 44.606 / 44.747 | 48.045 / 47.809 |
| B-only, `vmcnt(2)` | 45.178 / 45.202 | 47.872 / 48.009 |

Partial ping-pong is therefore closed for this geometry. It preserves
occupancy, global traffic, and the proven p8 bank phase, but it does not remove
either barrier. The early LDS writes contend with the 32 fragment reads and
the extra wait threshold extends the dependency graph. Moving work out of the
handoff is not useful when it merely moves that work onto the compute path.
The ordinary default device instructions remain unchanged after normalizing
the generated HIP CUID and assembly comments.

### 2026-08-23: P8 `ds_load_2addr_b64` with shifted bases

The earlier hand-scheduled two-address LDS kernel was limited to p4 because
the instruction offsets are unsigned eight-bit values in eight-byte units. A
p8 row consumes six units, putting the fourth 16-row fragment at 288. The new
`tools/make-p8-read2.py` transform adds A/B bases shifted by 512 bytes, so that
fragment uses offsets 224--227. Both bases are computed once before the hot
loop. The resulting code object uses 121 VGPR, 22 SGPR, 18 KiB LDS, and zero
spills.

Two forms were tested:

| Form | TFLOPS | Same-pass p8 controls | Reported occupancy |
|---|---:|---:|---:|
| Paired LDS reads and stores | 44.468 / 44.428 | 47.914 / 47.991 | 3 blocks / 24 waves |
| Paired reads, native b128 stores | 43.442 / 42.041 | 44.932 / 45.055 | 3 blocks / 24 waves |

All four outputs reproduced the full rocBLAS error tuple. The second bracket
ran at a visibly lower package state, so its absolute TFLOPS are not compared
to the first bracket; its own opening and closing controls still reject the
read-only form.

This experiment removes the encoding ambiguity and supplies another occupancy
counterexample. P8 native `ds_load_b128` is better than two independently
addressed 64-bit halves even when the latter reports eight more active waves.
Native refill stores do not rescue it, isolating the regression to the paired
read schedule rather than the store handoff. Do not revisit the p8 offset
limit without a new LDS instruction or a different physical fragment layout.

A follow-up tried to collapse the vector-offset MUBUF form's two independent
pointer increments into one dual-issue VALU instruction. The gfx1151 assembler
accepts the existing `v_dual_mov_b32 :: v_dual_add_nc_u32` pair but rejects
`v_dual_add_nc_u32 :: v_dual_add_nc_u32` as unsupported. Dual-issue slots are
opcode-pair constrained; this recurrence cannot be compressed unchanged and
was rejected before GPU use.

### 2026-08-23: Progressive VMEM-to-LDS commit after the barrier

The p8 leader issues three cooperative refill operations in the order A0, A1,
B. Its generated loop contained two consecutive full `vmcnt(0)` waits before
the overwrite barrier: one from the explicit source fence and another inserted
by LLVM. Moving only the source fence did not move the compiler wait, so the
experiment required a deterministic assembly transform.

`tools/patch_progressive_commit_asm.py` replaces that handoff with:

```asm
s_barrier
s_waitcnt vmcnt(2)
ds_store_b128  ; A0
s_waitcnt vmcnt(1)
ds_store_b128  ; A1
s_waitcnt vmcnt(0)
ds_store_b128  ; B
s_waitcnt vmcnt(0) lgkmcnt(0)
s_barrier
```

Program order makes each threshold unambiguous: `vmcnt(2)` guarantees the
oldest A0 load, `vmcnt(1)` guarantees A1 as well, and `vmcnt(0)` guarantees B.
The first barrier still prevents overwriting LDS while any wave consumes the
old tile; the final LDS wait and second barrier still publish the complete new
tile. The result assembles at 118 VGPR, 22 SGPR, 18 KiB LDS, and zero spills,
and reports two blocks/16 waves.

The full rocBLAS-reference tuple passed in every run. A short bracket produced
48.132/48.411 TFLOPS versus 48.158/48.193 controls. A longer interleaved
`C,X,C,X,C,X,C,X,C` screen produced controls of 47.785, 47.861, 47.734,
47.778, and 47.525 TFLOPS and candidates of 47.970, 48.124, 47.908, and
47.789. Pairing each candidate with its immediately preceding control gives
47.948 versus 47.790 TFLOPS, **+0.33%**. The closing control documents the
package drift but is not substituted for a candidate's preceding control.

Retain progressive commit as a composable scheduling primitive, not a
promotion. It is a small repeatable improvement with no resource change, on
the same scale as scalar-offset MUBUF recurrence.

The combined transform then passed the exactness gate at the same 118 VGPR,
22 SGPR, 18 KiB LDS, zero spills, and two reported blocks/16 waves. The short
bracket was:

| Form | TFLOPS | Average |
|---|---:|---:|
| P8 control | 48.184 / 48.332 | 48.258 |
| Progressive only | 48.599 / 48.323 | 48.461 |
| Progressive + scalar-offset MUBUF | 48.768 / 48.635 | 48.701 |

A longer `C,X,P,X,C,X,P,X,C` screen confirmed rather than erased the signal:

| Form | TFLOPS | Average |
|---|---:|---:|
| P8 control | 47.867 / 47.752 / 47.648 | 47.756 |
| Progressive only | 47.855 / 47.769 | 47.812 |
| Progressive + scalar-offset MUBUF | 48.317 / 47.990 / 48.138 / 48.247 | 48.173 |

The combined form is +0.75% over progressive-only and +0.87% over the controls
in that pass. All nine runs reproduced the full rocBLAS tuple. This establishes
additivity and makes the combined schedule the next research base, but it does
not replace the 48.614-TFLOPS sustained record: the package state was lower and
no sample reached the 50-TFLOPS promotion gate.

### 2026-08-23: One-fragment B lookahead

The next transform copied the compiler's pipelined final-tile pattern into the
steady loop. An alternate eight-VGPR bank holds B1 while B0 is consumed; the
ordinary B bank receives B2 while B1 is consumed; then the alternate bank
receives B3 while B2 is consumed. Each pair of `ds_load_b128` operations is
therefore separated from its consumer by four WMMAs rather than an immediate
full wait.

The initial assembly failed the reference gate with normalized maximum error
1.291 and cosine similarity 0.813. The dependency proof was not the bug: the
transform redirected the first B3 WMMA before the interleaved global refill,
but left the three following WMMAs reading B2. After all four consumers were
redirected, every placement reproduced the full reference tuple. Do not quote
the failed binary's throughput; its only result is the fragment-liveness rule.

The corrected register-base sweep was:

| Alternate B base | VGPR | Reported occupancy | TFLOPS |
|---:|---:|---:|---:|
| 118 | 126 | 3 blocks / 24 waves | 45.419 |
| 120 | 128 | 3 blocks / 24 waves | 45.999 |
| 122 | 130 | 2 blocks / 16 waves | 45.203 |
| 124 | 132 | 2 blocks / 16 waves | 45.929 |

The combined-base controls reached 48.352 and 48.276 TFLOPS. Register placement
is visibly load-bearing: v120 beats v118 by 1.28% at the same reported 24 waves,
and v124 beats v122 by 1.61% at the same 16 waves. That is direct gfx1151
evidence for the register-assignment optimization suggested by VeriLocc.

It does not make B lookahead viable. The best placement remains 4.8% behind
the control, while v118/v120 demonstrate that 24 reported waves can be slower
than the 16-wave base. Front-loading two more LDS operations and reading WMMA
operands from an alternate register bank cost more than overlapping the three
load/wait gaps saves. Retain the placement sweep as an optimization method;
close this one-fragment lookahead dataflow.

### 2026-08-23: Register boundary phase sets a new sustained leader

The B-lookahead placement spread motivated a semantics-preserving register
experiment. `tools/shift_vgpr_boundary_asm.py` leaves the eight FP16
accumulator fragments in v1--v64 and shifts every register at v65 or above by
one common delta. Addresses, A/B fragments, refill values, instruction order,
and dependencies therefore remain mutually unchanged; only their physical
phase relative to the accumulators moves.

Odd deltas failed assembly before GPU use. The loop contains:

```asm
v_dual_mov_b32 v58, v57 :: v_dual_add_nc_u32 v73, 0x3000, v73
```

After an odd boundary shift both destinations are even, violating gfx1151's
requirement that one dual-VALU destination be even and the other odd. Replacing
the pair with scalar instructions would change the instruction stream, so the
clean sweep retained only deltas 2/4/6.

All three forms passed the full rocBLAS tuple with 22 SGPR, 18 KiB LDS, and no
spills:

| Boundary delta | VGPR | Reported occupancy | TFLOPS |
|---:|---:|---:|---:|
| 0 control | 118 | 2 blocks / 16 waves | 48.399 / 48.381 |
| 2 | 120 | 2 blocks / 16 waves | **49.439** |
| 4 | 122 | 3 blocks / 24 waves | 45.049 |
| 6 | 124 | 3 blocks / 24 waves | 46.025 |

Delta 2 stayed positive in a longer `C,X` screen. Candidate medians were
49.431/49.435/49.177/49.249 TFLOPS, averaging **49.323**. Their immediately
preceding controls were 48.307/48.181/48.147/48.134, averaging 48.192; the
gain is **+2.35%**. The closing control was 48.238 and documents drift.

The record qualification then ran five fresh processes with 20 warmups and
five timing blocks of 100 iterations per process:

| Process | Median ms | TFLOPS |
|---:|---:|---:|
| 1 | 2.799422 | 49.095 |
| 2 | 2.806014 | 48.980 |
| 3 | 2.805167 | 48.995 |
| 4 | 2.805672 | 48.986 |
| 5 | 2.798256 | 49.116 |

The average is **49.035 TFLOPS**, the floor is **48.980 TFLOPS**, and every
process reproduced normalized maximum error 0.018779343, RMS 0.035428338, and
cosine similarity 0.999977929. Opening and closing combined-base controls were
48.103/47.941 TFLOPS. Delta 2 is +2.11% over their average and +0.87% over the
former 48.614-TFLOPS sustained leader.

Promote the delta-2 phase as the new prepacked-contract research base. It does
not meet the 50-TFLOPS goal: the sustained gap is 1.97%. The delta-4/6 failure
despite 24 reported waves is equally important--this uplift is a physical
WMMA register-phase effect, not an occupancy result.

The next isolation kept the two-register gap but moved its boundary across all
safe cuts in the generated live ranges. This pass ran in a much lower package
state, so only same-pass deltas are meaningful:

| Shift boundary | TFLOPS | Delta vs 45.406 control midpoint |
|---:|---:|---:|
| v17 | 44.430 | -2.15% |
| v33 | 44.461 | -2.08% |
| v49 | 44.464 | -2.07% |
| v57 | 44.848 | -1.23% |
| v66 | 45.150 | -0.56% |
| v68 | 44.420 | -2.17% |
| v69 | 44.513 | -1.97% |
| v70 | 44.607 | -1.76% |

The v65 delta-2 controls opened at 45.510 and closed at 45.302 TFLOPS; do not
compare this throttled bracket's absolute values with the 49.035 record screen.
Every boundary candidate remained exact at 120 VGPR and 16 reported waves.
None improves v65, establishing that the winning group includes v65 itself and
all later address/fragment/refill registers. The near-neutral v66 result
isolates physical v65 placement as a small but measurable part of the gain.

### 2026-08-23: Fine-grained register permutations close as noise

The boundary result left a narrower question: can the 120-VGPR delta-2 kernel
improve further by changing only the assignment of equally sized live groups?
Two exact-involution assembly transforms answered it without changing the
instruction stream, live count, dependencies, or resource metadata.

The first transform swapped B's eight-register hot-loop bank with each A bank.
The second swapped epilogue-safe 16-register accumulator pairs. Accumulator
pairs involving v1 were rejected statically because the epilogue consumes the
contiguous `v[0:1]` range. Every built candidate used 120 VGPR, 22 SGPR,
18 KiB LDS, zero spills, and reproduced the full rocBLAS tuple.

| First bracket | TFLOPS |
|---|---:|
| Controls | 49.580 / 49.622 |
| B/A0 | 49.767 |
| B/A1 | 49.532 |
| B/A2 | 49.631 |
| B/A3 | 49.729 |
| Accumulators v17/v33 | 49.602 |
| Accumulators v17/v49 | 49.785 |
| Accumulators v33/v49 | 49.514 |

The apparent B/A0 and v17/v49 positives were then composed and re-bracketed:

| Composition bracket | TFLOPS |
|---|---:|
| Controls | 49.863 / 49.827 |
| B/A0 | 49.598 |
| Accumulators v17/v49 | 49.567 |
| B/A0 + accumulators v17/v49 | 49.552 / 49.886 |
| B/A3 + accumulators v17/v49 | 49.758 |

The signs reversed and the best combined average stayed below the control
midpoint. Close these same-width group swaps as run-order/package noise. Keep
the unpermuted delta-2 allocation as the research base.

### 2026-08-23: Paperclip scheduling refresh and WMMA row-order screen

The latest Paperclip search added current AMD scheduling evidence to the
notebook. HipKittens' eight-wave ping-pong and four-wave interleave results are
the closest architectural match: AMD's static per-wave register allocation
makes dedicated producer waves expensive, while explicit issue order and
fine-grained overlap remain valuable. Twill and Tawa provide formal
software-pipeline/producer-consumer models, but their TMA, WGMMA, and mbarrier
mechanisms are NVIDIA-specific. FIBER's shared-register solution requires new
hardware. The HIP autotuning study explains why gfx1151 needs empirical
searches: AMD optima can be unusually sharp. tritonBLAS' tile/cache model is
useful for grid-shape reasoning, but the square 4096 workload already has a
fully occupied 2048-workgroup grid.

That evidence motivated `tools/reorder_hot_wmma_rows_asm.py`. It leaves the
progressive first WMMA fragment and all refill instructions fixed, then
permutes the three later four-row WMMA issue groups. The final group must issue
row 0 first because the immediately following global refill overwrites A0.
The optional group-specific form is generated by
`build-wmma-row-order-isolation.sh`. All assembled forms are 120 VGPR,
22 SGPR, 18 KiB LDS, zero spills, and exact under the full rocBLAS tuple.

The initial exploratory bracket was interrupted for an Ember release and ran
at a lower package state. Controls were 44.954/44.808 TFLOPS; orders
0132/0213/0231/0312/0321 reached 44.853/45.231/44.855/45.348/45.274.
The apparent 0312 gain is +1.04% over the same-pass control midpoint, not a
new record. Do not compare its absolute throughput with the 49.035-TFLOPS
five-process qualification. Re-bracket the isolated groups when the GPU is
available, then require five fresh processes before promotion.

The follow-up isolation bracket later ran with identical 20-warmup,
five-block, 20-iteration settings and full validation on every process.
Opening/closing controls were 49.295/49.176 TFLOPS. Group-specific orders
g1/g2/g3 reached 49.285/49.329/49.078; g1+g2, g1+g3, and g2+g3 reached
49.006/49.091/49.095. The all-group 0312 schedule reached **49.401 TFLOPS**.
All candidates stayed at 120 VGPR, 22 SGPR, 18 KiB LDS, zero spills, and the
exact normalized error tuple. The all-group result is a useful short-screen
candidate, not a new record: its approximately 0.34% edge over the two-control
midpoint is not separated from package/order drift and it has no fresh-process
50-TFLOPS qualification.

The subsequent five-fresh-process screen used 20 warmups and five 100-iteration
timing blocks per process. Delta-2 base medians were 49.173/49.162/48.910/
48.798/48.694 TFLOPS; all-group 0312 medians were 49.066/48.809/48.515/48.834/
48.830. Full validation passed every process, but the row-order candidate lost
to its immediately preceding control in all five pairs. Close row-order
permutations as noise and retain the unpermuted delta-2 phase.

### 2026-08-24: paired-K IU4 pipeline

The first positive new architecture after the FP16 delta-2 screens is an
isolated paired-K pipeline for the linear W4A4 contract. With
`-DIU4_PAIR_K=1`, the kernel keeps two K16 slices in four rotating LDS slots,
loads the next pair while the current pair executes, and synchronizes once per
pair boundary. It uses 103 VGPR, 18 SGPR, 8,192 bytes of LDS, and no spills.

At 4096 cubed, five interleaved fresh candidate/control pairs measured
90.527/90.208/89.381/90.599/90.129 INT4 TOPS for the paired candidate (90.169
average, 89.381 floor), versus 84.961/84.928/84.942/84.829/85.138 for the
one-slice control (84.960 average). Every candidate and control produced zero
mismatches over all 16,777,216 INT32 outputs. This exceeds the 50-operations/s
threshold only under the separate integer contract; it does not alter the
49.035-TFLOPS FP16 leader. The four-slot/pair-boundary schedule is nevertheless
a useful architecture to revisit with a smaller FP16 tile or a different
producer/consumer partition.

A 256x256 FP16 supertile was tested as the first new fragment/dataflow
architecture. It preserves the per-wave fragment footprint and reuses A across
twice as many N columns, compiling at 119 VGPR, 22 SGPR, and 24 KiB LDS. The
16-wave block permits only one resident block per CU, however; despite passing
the exact output gate it reached 37.486 TFLOPS. The result confirms that A reuse
alone cannot replace the current two-block 256x128 schedule.

The transposed 128x256 control retained two-block residency and passed exact
validation at 45.930 TFLOPS with 119 VGPR and 18 KiB LDS. A textual conversion
of its global refills to scalar-offset MUBUF produced an apparent >50-TFLOPS
timing, but normalized error was 1.34 with near-zero cosine. This is a useful
warning: the transposed source has a different pointer/address contract and
cannot inherit the 256x128 hand assembly by register substitution.

The corresponding FP16 transfer screen found a hard resource/layout boundary.
The supported 128x128 four-wave block-prepacked mapping compiled at 153 VGPR
and 12 KiB LDS, passed the exact output check, and measured 40.661 TFLOPS. A
forced eight-wave mapping faulted before validation because the existing WMMA
fragment contract is not valid for that geometry. A four-slot paired-K design
at 256x128 would consume approximately 48 KiB LDS per block, eliminating the
two-block occupancy of the 49-TFLOPS delta-2 kernel. More LDS buffering alone is
therefore not the route to 50 FP16 TFLOPS; a new fragment partition is required.

A follow-up composition sweep shifted the B LDS base of delta-2 by +4, +8, and
+16 bytes. All forms were exact at 120 VGPR, but measured 38.059, 38.051, and
48.520 TFLOPS in the same lock window. The first two shifts expose a severe LDS
bank-phase penalty and the 16-byte shift remains slower than delta-2. Register
phase and LDS phase must therefore be optimized jointly rather than composed
from separately favorable screens.

The retained delta-2 binary was requalified on the current host in five fresh
processes. Exact medians were 49.052, 48.828, 48.846, 48.724, and 48.680
TFLOPS (48.826 average). This confirms the run-to-run package sensitivity and
leaves the prior 49.035 qualification as the stronger record; no process met
the 50-TFLOPS gate.

## 2026-08-24: 256x64 fragment geometry

A new 256x64 block geometry was implemented with an explicit four-vector A
loader and four resident blocks per CU. After fixing the 32-lane wave mapping,
it passed exact validation at 42.962 TFLOPS. Splitting N in half duplicates A
traffic enough to erase the occupancy benefit, so this geometry is closed.

The hand assembly’s scalar register placement was then shifted by aligned
four- and eight-register gaps. Both variants passed the complete exactness
tuple without changing VGPR occupancy. The +4 phase averaged about 48.73
TFLOPS; the +8 phase averaged 49.053 with a 48.880 floor. Neither separates
from package-sensitive delta-2 controls, so no SGPR phase is retained.

The A-side LDS base was then shifted by +8 and +16 bytes in every initial and
refill access. Both forms were exact. +8 fell to 35.843 TFLOPS; +16 averaged
about 48.997 TFLOPS across five processes. The bank phase is load-bearing but
not a standalone source of headroom, so the A-LDS phase branch is closed.

Combining the operand phases was also tested: every A and B LDS access moved by
16 bytes. The candidate stayed exact with unchanged occupancy, but five
processes averaged about 48.726 TFLOPS (48.649--48.822). A/B phase interaction
does not provide headroom, so the combined phase is not retained.

Finally, paired A LDS requests were reordered by bank offset without changing
their destinations or waits. The candidate was exact but averaged about
48.629 TFLOPS across five processes. This issue-order variant is closed.

An A-fragment register permutation then swapped the A0/A1 groups together with
all loads, stores, and WMMA uses. It remained exact at unchanged occupancy, but
averaged about 48.855 TFLOPS across five processes. Physical A-bank placement
alone is not a route to 50.

The complementary A2/A3 group swap was exact but averaged about 48.752 TFLOPS
across five processes. Both A-fragment pairings are now closed.

The final tile’s B0/B1 register groups were also exchanged with all matching
loads and WMMA uses. It remained exact but averaged about 48.711 TFLOPS across
five processes, so final-tile B placement is closed.

The complementary 256x192 tile was not a valid benchmark geometry: 192 does
not divide 4096, and its edge workgroup faulted before validation. No timing is
recorded; non-divisible tiles require a separate bounds-safe harness.

An independent loop-counter decrement was moved into the first WMMA issue
window as a SALU/VALU overlap experiment. It stayed exact at the same resource
tuple, but five processes measured 48.777, 48.740, 48.669, 48.624, and 48.788
TFLOPS (48.720 average, 48.624 floor). The 49.094 short result was noise; this
schedule change is closed.

The K32-per-stage source specialization was initially run with a host-packer
mismatch: the kernel used K32 while `RECORD_K_SLICES` remained 1. That invalid
44.350-TFLOPS result failed exactness. Rebuilding both sides for K32 repaired
the output and measured 44.289 TFLOPS at two-block occupancy. The dedicated K2
ring with correct host packing failed exactness (normalized error 1.365675535,
cosine -0.000019926) behind a 42.198-TFLOPS timing; its earlier 41.345-TFLOPS
“exact” result used mismatched K16 packing and is invalid. Correct packing does
not make K32 staging competitive.

The corrected K2 no-explicit-VMEM-wait form was also exact but reached 44.321
TFLOPS versus 44.289 for the control. The K32 handoff cost remains load-bearing.

The alternate slice-major K32 host packing was also tested. It ran at 44.424
TFLOPS but failed exactness with normalized error 1.356285863 and cosine
-0.000246312. The K2 address contract is block/K-major; slice-major packing is
not compatible.

The delta-2 control was rechecked on the current host at 48.911 TFLOPS with
exact output (20 warmups, 100 iterations per timing block). It remains inside
the established qualification band and is not a new record.

The half-word LDS swizzle companion was exact at two-block occupancy but
reached only 38.402 TFLOPS. This layout permutation is also closed; further
progress needs a new packed producer rather than another swizzle.

An early-B prefetch probe moved the next global B vector into v120:v123 before
the second WMMA group. The initial image illegally declared 120 VGPR and
showed an invalid 51.372-TFLOPS timing. A 128-VGPR image with reordered VMEM
completion and copy-back still failed exactness at 49.575 TFLOPS. The apparent
50+ result is therefore rejected; the live-range/register contract must be
redesigned before this producer idea can be revisited.

After auditing the transformation script, the early-B image was regenerated
with the relocated load verified present, 128-VGPR metadata, and the intended
copy/wait sequence. It still failed exactness at 49.886 TFLOPS, confirming that
the producer idea itself—not just the first script artifact—is invalid.

The complementary early-A high-register staging image hung before producing
occupancy or validation output and was terminated. It has no timing result and
is rejected as a synchronization/launch failure rather than a performance
candidate.

The hot-loop global-load clause was widened to include A0, A1, and the B
prefetch together. It remained exact at unchanged occupancy, but five
processes measured 48.961, 48.858, 48.840, 48.854, and 48.946 TFLOPS (48.892
average, 48.840 floor). The isolated 49.103 result was noise; clause grouping
does not close the gap.

An A-load width fusion using `buffer_load_b256` was attempted, but gfx1151
rejected the vector instruction at assembly (only scalar `s_buffer_load_b256`
was suggested). No image or timing exists; this instruction-level path is
closed by the ISA.

The source `warp_tile_m=2` ownership variant was also screened. Four M-waves
derived a 128x128 tile and failed exactness at 23.282 TFLOPS; eight M-waves
restored 256x128 but failed launch before validation. No candidate is retained.

The opposite `warp_tile_m=8` geometry was also tested. Two N-waves gave a
non-finite output behind an invalid 86.892-TFLOPS timing; four N-waves repaired
exactness but reached only 47.590 TFLOPS. The larger M stripe is not a viable
route to 50 TFLOPS.

The source paired-B staging path was also screened on 256x128. It remained
exact at unchanged occupancy but reached only 47.698 TFLOPS, so B pairing is
closed as a standalone optimization.

Forcing 64-bit native LDS fragment loads was also exact at unchanged occupancy
but reached only 41.618 TFLOPS. Narrowing the native load width is closed as a
performance path.

The streamed-B barrier toggle was also exact but reached only 44.802 TFLOPS
with three active blocks/24 waves per CU. Additional scheduler barriers are
counterproductive, so this synchronization path is closed.

Resizing the source `c_n` and packed-path `a_frag` arrays for eight M-fragments
removed the obvious out-of-bounds state, but the image still produced
non-finite output behind an invalid 87.047-TFLOPS timing. The remaining
fragment/epilogue assumptions make this ownership geometry unsuitable.

Complementary asymmetric A/B padding pairs (4/12, 12/4, 2/14, 14/2, 6/10,
10/6) all stayed exact but measured only 25.755--27.389 TFLOPS. The stride
phases cannot be decoupled without upsetting the WMMA/LDS access pattern, so
this layout family is closed.

An explicit VMEM wait before copying the high-register B vector was also
tested; it remained wrong at 49.274 TFLOPS. The early-B image is closed rather
than treated as a performance result.

### 2026-08-24: MAC-cluster priority experiment

The next new schedule hypothesis used the gfx11 `s_setprio` control around the
existing hand-scheduled WMMA cluster. `tools/patch_setprio_asm.py` inserts a
priority raise at the start of the steady-state and final-tile compute loops,
then restores normal priority before the LDS producer/consumer handoff. This
preserves the exact delta-2 packed contract and its 120-VGPR/22-SGPR/18-KiB
resource tuple.

The assembled image was rejected by the gfx1151 runtime before the numerical
gate: every one of five isolated launches returned `invalid device function`
from `hipOccupancyMaxActiveBlocksPerMultiprocessor`. The source macro variant
was loadable and exact, but its 41.84--42.26 TFLOPS results are the unscheduled
source baseline, not evidence for the hand-scheduled leader. This closes
priority control as a route to 50 and reinforces that an assembler-accepted
instruction is not necessarily a usable gfx1151 runtime instruction.

The remaining late-refill double-buffer branch was then built as a new packed
dataflow: `WMMA_BP_DOUBLE_BUFFER_LATE=1` retains both complete operands in
inactive/active LDS buffers and moves the refill later into the WMMA cluster.
It passed the full exactness tuple in every isolated launch, but the second LDS
tile reduced residency to one block/eight waves per CU. Five medians measured
38.987, 39.192, 38.978, 39.197, and 38.856 TFLOPS (39.042 average, 38.856
floor). The architecture is numerically sound but occupancy-bound and is
closed for the 4096-square target.

The synchronization follow-up then tested a split producer/consumer barrier.
ROCm 7.14 rejected `s_barrier_signal` and `s_barrier_wait` for gfx1151 during
assembly, confirming that this hardware has no usable split-barrier path in
the current toolchain. A related wait-order build (`WMMA_BP_WAIT_AFTER_BARRIER`)
placed the VMEM wait after the workgroup barrier; it was loadable and exact, but
five medians were 47.746, 47.619, 47.633, 47.385, and 47.335 TFLOPS (47.543
average, 47.335 floor). The original wait-before-barrier schedule remains the
control.

The hybrid A producer was then screened with grouped A loads and a late commit
to the inactive LDS buffer. The exact 256x128 packed image retained two
blocks/16 waves, but five medians measured 44.551, 44.539, 44.333, 44.391, and
44.246 TFLOPS (44.412 average, 44.246 floor). This distinct one-sided
producer/consumer schedule is numerically sound but below the retained leader;
its extra A handoff work closes the hybrid-A route for the square shape.

The late-B1 specialization was also built with its required 128x128,
single-buffer ownership geometry. It passed the exact reference tuple in all
five launches and reported four blocks/16 waves, but medians were 41.166,
40.840, 40.764, 40.721, and 40.902 TFLOPS (40.879 average, 40.721 floor).
The smaller tile's traversal/refill cost dominates, so late-B1 is closed as a
route to the 50-TFLOPS target.

The hand-scheduled fixed-shape epilogue was then explored. Removing only the
late N-fragment edge compares produced an apparent 49.117 TFLOPS timing, but
the full output was wrong (normalized error 0.848721961, RMS 2.167147235,
cosine 0.913690612). Removing the exec masks and branches as well yielded an
invalid gfx1151 image before validation. The edge predicates are therefore
load-bearing in the generated schedule; this fixed-tile specialization is
closed until the epilogue is redesigned rather than mechanically stripped.

The source redesign was then implemented behind `WMMA_BP_FULL_TILE_STORE=1`.
It keeps the ordinary exec state and fragment mapping but removes per-element
bounds checks for the exact 256x128 record geometry. Five launches were exact
at 47.749, 47.805, 47.555, 47.730, and 47.558 TFLOPS (47.679 average, 47.555
floor). The source full-tile epilogue is safe but slower than the
hand-scheduled delta-2 leader, so it is closed as a standalone performance
route.

A fresh ROCm Compute Profiler pass then measured the retained delta-2 image in
separate counter passes. `SQ_BUSY_CYCLES_avr / GRBM_GUI_ACTIVE` was 0.9904;
normalized wave-cycle counters were approximately 17.2% barrier wait, 5.7%
LDS-instruction wait, and 8.6% counter wait. These counters are not timing
denominators, but they sharpen the next design target: hide or remove the two
workgroup handoffs without increasing the 18-KiB LDS footprint or 120-VGPR
allocation. Raw files are retained at `/root/wmma-results/profile-delta2-new/`.

The next producer/consumer experiment assigned two waves to global-to-LDS
production and four to WMMA consumption. Its first launch exposed a harness
mistake: the host still launched five waves instead of the required six and
deadlocked waiting for a missing consumer. Rebuilding with
`RECORD_PRODUCER_WAVES=6` produced exact output at three blocks/18 waves, but
three fresh processes measured only 13.070, 12.746, and 12.800 TFLOPS (12.872
average). The two-producer ring is therefore correctly closed as a throughput
regression; the initial deadlock was not a kernel conclusion.

The wide producer/consumer variant was then launched correctly with nine waves
(one producer and eight consumers). It passed exactness but reached only 6.517
TFLOPS in a 5-warmup/5-iteration screen at three blocks/27 waves per CU. The
wide ring is closed as a throughput regression without a longer promotion run.

The next scalar-control probe removed the K-loop `s_cmp_eq_u32 s6, 0`, relying
on the preceding decrement's SCC for the back-edge branch. Although the image
assembled, gfx1151 rejected it as `invalid device function` before occupancy
and validation. The explicit loop compare is retained as a required code-object
boundary; no throughput result is recorded.

The hand epilogue address path was then changed to a 64-bit recurrence for the
first eight constant-stride stores, using two otherwise-unused VGPRs and a
carry-aware VOP3 increment. Both syntactic carry-in orderings assembled but
were rejected by gfx1151 as `invalid device function` before occupancy. The
original per-store address reconstruction remains required; no performance
result is recorded.

The synchronization follow-up moved the first refill `s_waitcnt vmcnt(2)` ahead
of the publish barrier to overlap VMEM retirement with wave convergence. The
gfx1151 loader rejected the assembled image as `invalid device function` before
occupancy, so no timing or correctness result exists. The original
barrier-then-wait ordering remains required.

The complementary one-sided B producer was then built with
`WMMA_BP_HYBRID_B_PINGPONG=1`, leaving A in the active buffer. It stayed exact
at two blocks/16 waves, but five medians were 45.090, 45.107, 44.805, 44.976,
and 44.742 TFLOPS (44.944 average, 44.742 floor). B-side ping-pong does not
recover the hand-scheduled leader's gap and is closed as an independent path.

The counter-guided follow-up then relaxed one `lgkmcnt` threshold at a time in
the hand-scheduled delta-2 loop. Lowering 6→5, 4→3, or 2→1 was bit-exact and
gave attractive short 10/10 timings of 49.930, 49.807, and 49.858 TFLOPS.
Long interleaved fresh-process tests did not sustain a promotion: the 2→1
candidate averaged 48.807 versus 48.565 for controls, and the 6→5 candidate
averaged 48.952 versus 48.893. Applying all three relaxations together fell
to 49.597 in the short screen. The original 6/4/2 dependency ladder remains
the safe schedule; these isolated wait edits are closed as noise-sensitive
rather than a new architecture.

For completeness, the complementary normal-layout 128x256 ownership geometry
was rebuilt with two M waves and four N waves. It retained two-block/16-wave
occupancy and the exact packed output tuple, but measured 46.809 TFLOPS in a
10/10 screen. This confirms that doubling N while halving the M stripe does not
recover the leader; the 256x128 ownership geometry remains the only viable
source shape in this family.

The epilogue was then probed for direct vector stores. Packing the low halves
of two neighboring accumulator registers with `v_pack_b32_f16` and replacing
two half stores with one dword store assembled and loaded, but failed the full
reference (normalized maximum error 0.767344810, RMS 0.668158748, cosine
0.992118715). The hardware fragment register order is not directly contiguous
in output memory; no timing from this invalid image is retained.

The same hand schedule was then regenerated with LDS swizzle 8 and 32. Both
variants were exact at unchanged 120-VGPR/two-block occupancy and produced
49.410/49.439 TFLOPS in short screens. The swizzle-32 form lost in the fresh
interleaved comparison (48.771 average versus 48.994 for delta-2 controls),
so alternate bank phases are closed and the default swizzle 16 remains the
research base.
Zero A/B LDS padding was then compiled as a source control. The hand patcher
correctly refused to apply its p8-specific descriptor rewrite, so no invalid
hand image was timed. The source kernel itself was exact at two-block/16-wave
occupancy but reached 45.272 TFLOPS in a 10/10 screen, closing zero padding
without weakening the p8 contract.

Keeping A at p8 while removing B padding was tested as a mixed-bank control.
The p8-specific hand patch correctly refused the changed descriptor pattern,
and the source-only A8/B0 image was exact but reached 44.993 TFLOPS in a 10/10
screen. This mixed layout is closed until a separate hand descriptor generator
exists.

The corrected DPP vector epilogue was also combined with the delta-2 hand
schedule. Its source allocation is 117 VGPR rather than the hand image's 118,
and the handoff patch has no matching descriptor. Adjusting the register-phase
tool to that allocation produced dual-VALU VGPR-bank conflicts rejected by the
gfx1151 assembler. This combined path produced no loadable image and is not a
performance result.

The DPP source was also swept with a delta-4 register phase, which preserves
the four-bank dual-VALU pairing that delta-2 violates. Safe boundaries from
33 through 81 all assembled and stayed exact, but the resulting 121-VGPR
images admitted three blocks/24 waves and reached only 41.106--41.383 TFLOPS
in 5/5 screens. This closes DPP register-phase shifts as a route to 50.

The mapper prologue was then specialized for the exact 16x32 record grid.
Direct row-major shifts cut the integer setup and stayed exact, but fresh
interleaved comparisons averaged 48.536 TFLOPS versus 48.773 controls after a
49.596 short screen. A hand reconstruction of the source column XOR-snake map
was not exact (normalized error 1.000000), showing that the cache traversal
contract is not captured by the naive scalar formula. Fixed-grid mapping is
closed without a verified equivalent traversal.

The known `cu_5x8_record_mapping` was then evaluated as a different cache
traversal. The complete source mode-5 image was exact at 46.933 TFLOPS. A
prologue-only splice into delta-2 failed because mode 5 changes the downstream
pointer/SGPR contract; a source mode-5 register shift was exact but reached
38.857 TFLOPS at 121 VGPR/three blocks. The traversal needs a full hand
regeneration, not a prefix transplant.

The exact-grid mapper was also tested in the complementary fixed
column-major order. It retained the hand schedule and exact output tuple but
fell to 44.403 TFLOPS in a 5/5 screen, so the row/column traversal is not a
free scheduling choice. The experiment is closed.

The IU4 branch remains a separate architecture rather than an FP16 shortcut.
With `-DIU4_PAIR_K=1`, the prepacked linear W4A4 GEMM held 90.945 INT4 TOPS on
average across five fresh candidate/control pairs (90.105--92.298), all exact
over the full output. The one-slice controls averaged 84.560 TOPS. The paired
K16 residency removes one publication/barrier boundary per two slices, but
ROCMFP4 codebook weights still cannot be sent directly to IU4 without a new
quantization and scaling contract.

The paired-K ring was generalized to four K16 slices (`IU4_PAIR_K=2`) in eight
rotating LDS slots. The second fresh five-process bracket measured 94.044 INT4
TOPS on average (93.445--94.700), exact in every process, against 91.244 for
the two-slice control. A separate earlier bracket had one 84.765 TOPS outlier;
it is retained as noise evidence, not as the promoted number. Four-slice IU4
is the strongest integer-WMMA architecture so far, while the FP16 50-TFLOPS
gate remains open and the ROCMFP4 codebook contract is unchanged.
