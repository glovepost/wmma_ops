// ============================================================================
// WMMA DEVICE HELPER FUNCTIONS
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
// 
// Architecture-specific device helper functions for WMMA kernels
// ============================================================================

#ifndef WMMA_DEVICE_HELPERS_HPP
#define WMMA_DEVICE_HELPERS_HPP

#include "wmma_xor_swizzle.hpp"  // For type definitions (half8, etc.)

// ============================================================================
// GFX1151-SPECIFIC INTRINSICS AND ASSEMBLY HELPERS
// These are architecture-specific and NOT portable
// ============================================================================

// Direct vectorized load to LDS via registers (ROCm 6.x compatible)
// RDNA3 can dual-issue global_load + WMMA, so this still benefits from interleaving
__device__ __forceinline__ void load_half8_to_lds(
    const __half* __restrict__ global_ptr,
    __half* lds_ptr,
    bool is_valid
) {
    if (is_valid) {
        half8 data = *reinterpret_cast<const half8*>(global_ptr);
        *reinterpret_cast<half8*>(lds_ptr) = data;
    }
}

// Hardware data prefetch for global memory
// Note: s_prefetch_data is not available on RDNA3.5, use software prefetch
__device__ __forceinline__ void hw_prefetch_global(const void* ptr, int size_bytes) {
    // Use compiler builtin for prefetch
    __builtin_prefetch(ptr, 0, 3);
}

// Explicit memory fence for LDS
__device__ __forceinline__ void lds_fence() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// Explicit memory fence for VMEM
__device__ __forceinline__ void vmem_fence() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// Combined fence
__device__ __forceinline__ void full_fence() {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

// ============================================================================
// VECTORIZED B TRANSPOSE WITH REGISTER STAGING
// Replaces scalar scatter B transpose to eliminate bank conflicts and UB
// ============================================================================

// Shuffle a _Float16 within an 8-lane subgroup (lanes 0-7, 8-15, 16-23, 24-31)
// Uses explicit lane calculation since AMD's __shfl width parameter may not work as expected
__device__ __forceinline__ _Float16 shfl_f16_width8(_Float16 v, int srcLane)
{
    uint16_t u16 = __builtin_bit_cast(uint16_t, v);
    uint32_t u32 = (uint32_t)u16;
    // Calculate actual source lane: same subgroup base + srcLane within subgroup
    int lane_id = threadIdx.x & 31;  // lane within warp
    int subgroup_base = lane_id & ~7;  // base of 8-lane subgroup (0, 8, 16, or 24)
    int actual_src = subgroup_base + (srcLane & 7);
    uint32_t r32 = __shfl(u32, actual_src);  // full-width shuffle to explicit lane
    uint16_t r16 = (uint16_t)r32;
    return __builtin_bit_cast(_Float16, r16);
}

// Vectorized B transpose: load half8 from GMEM, transpose via shuffles, store half8 to LDS
// B is row-major [K, N] in GMEM, stored as column-major B_lds[N][K] in LDS
// 
// Parameters:
//   B_lds_base: pointer to B_lds[0][0] for current buffer
//   B: global B matrix pointer
//   N, K: matrix dimensions
//   block_n: starting N index for this block
//   k_offset: starting K index for this tile
//   tid: thread ID within the 128 threads assigned to B loading
//   lds_stride: stride in K dimension for LDS (typically BLOCK_K + LDS_PAD)
//   block_n_size: BLOCK_N (typically 64 or 128)
//   block_k_size: BLOCK_K (typically 16)
//
// Thread mapping for BLOCK_N=64, BLOCK_K=16 (128 threads):
//   - 8 n-vectors (64/8) × 2 k-groups (16/8) × 8 lanes = 128 threads
//   - Each 8-lane subgroup transposes one 8×8 tile
template<int BLOCK_N, int BLOCK_K, int LDS_STRIDE>
__device__ __forceinline__ void load_B_tile_vectorized(
    __half* __restrict__ B_lds_base,
    const __half* __restrict__ B,
    int N, int K,
    int block_n, int k_offset,
    int tid
) {
    using half8_t = _Float16 __attribute__((ext_vector_type(8)));
    
    // Total vectors to load: BLOCK_K rows × (BLOCK_N/8) vectors per row
    // For BLOCK_N=64, BLOCK_K=16: 16 × 8 = 128 vectors
    // For BLOCK_N=128, BLOCK_K=16: 16 × 16 = 256 vectors
    constexpr int NVECS = BLOCK_N / 8;
    constexpr int TOTAL_VECS = BLOCK_K * NVECS;
    
    // Only threads with tid < TOTAL_VECS participate
    if (tid >= TOTAL_VECS) return;
    
    // Map thread to (b_k, b_n_vec): tid = b_k * NVECS + b_n_vec
    const int b_k = tid / NVECS;
    const int b_n_vec = tid % NVECS;
    const int b_n = b_n_vec * 8;
    
    const int k = k_offset + b_k;
    const int n_gmem = block_n + b_n;
    
    // Load 8 consecutive N elements from B[k][n_gmem..n_gmem+7]
    half8_t b_vec = {0,0,0,0,0,0,0,0};
    if (k < K && (n_gmem + 7) < N) {
        b_vec = *reinterpret_cast<const half8_t*>(B + k * N + n_gmem);
    }
    
    // Store transposed: B_lds[n][k] for n = b_n..b_n+7
    // Use union to safely extract elements from vector type
    union { half8_t vec; _Float16 elems[8]; } u;
    u.vec = b_vec;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        B_lds_base[(b_n + i) * LDS_STRIDE + b_k] = *reinterpret_cast<__half*>(&u.elems[i]);
    }
}

// Simplified version for common 128-thread, BLOCK_N=64, BLOCK_K=16 case
// Only threads 0-127 participate; threads >= 128 are no-ops
// Uses LDS for transpose instead of shuffles to avoid warp alignment issues
__device__ __forceinline__ void load_B_tile_vec_64x16(
    __half* __restrict__ B_lds_base,
    const __half* __restrict__ B,
    int N, int K,
    int block_n, int k_offset,
    int tid,
    int lds_stride
) {
    // Only first 128 threads participate (16 groups of 8)
    if (tid >= 128) return;
    
    using half8_t = _Float16 __attribute__((ext_vector_type(8)));
    
    // Thread mapping: 128 threads load 128 vectors (16 K rows × 8 N vectors per row)
    const int b_k = tid >> 3;      // 0..15 (K row)
    const int b_n_vec = tid & 7;   // 0..7 (which N vector)
    const int b_n = b_n_vec * 8;   // N column base (0, 8, 16, ..., 56)
    
    const int k = k_offset + b_k;
    const int n_gmem = block_n + b_n;
    
    // Load 8 consecutive N elements from B[k][n_gmem..n_gmem+7]
    half8_t b_vec = {0,0,0,0,0,0,0,0};
    if (k < K && (n_gmem + 7) < N) {
        b_vec = *reinterpret_cast<const half8_t*>(B + k * N + n_gmem);
    }
    
    // Store transposed: B_lds[n][k] for n = b_n..b_n+7
    // Each element b_vec[i] goes to B_lds[b_n + i][b_k]
    // Use union to safely extract elements from vector type
    union { half8_t vec; _Float16 elems[8]; } u;
    u.vec = b_vec;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        B_lds_base[(b_n + i) * lds_stride + b_k] = *reinterpret_cast<__half*>(&u.elems[i]);
    }
}

#endif // WMMA_DEVICE_HELPERS_HPP

