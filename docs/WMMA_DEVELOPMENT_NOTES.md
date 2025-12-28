# Code Organization Plan for wmma_gemm.hip

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
