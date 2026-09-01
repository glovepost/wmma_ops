// ============================================================================
// WMMA GEMM OPTIMIZED KERNEL V4
// Fixed B matrix LDS layout to match load_matrix_sync_lds_b_transposed expectations
//
// Key optimizations:
// 1. Larger tiles: 128×128
// 2. 4×2 warp layout with 2×4 warp tiling: 8 WMMA tiles per warp
// 3. Cooperative loading with proper LDS layouts
// 4. B stored column-major in LDS: B_lds[N][K] for transposed fragment loading
// ============================================================================

#ifndef WMMA_KERNEL_LARGETILE_HPP
#define WMMA_KERNEL_LARGETILE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

typedef _Float16 half8_opt __attribute__((ext_vector_type(8)));

struct OptConfig {
    static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, WARP_SIZE = 32;
    static constexpr int WARPS_M = 4, WARPS_N = 2, NWARPS = 8;
    static constexpr int WARP_TILE_M = 2, WARP_TILE_N = 4;
    static constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16;
    static constexpr int LDS_STRIDE_A = BLOCK_K + 8;  // 24 - stride in K dimension for A
    static constexpr int LDS_STRIDE_B = BLOCK_K + 8;  // 24 - stride in K dimension for B (column-major)
    static constexpr int NUM_THREADS = 256, HALF_BLOCK = 128;
};

#ifndef WMMA_OPT_MIN_BLOCKS_PER_CU
#define WMMA_OPT_MIN_BLOCKS_PER_CU 2
#endif

#if WMMA_OPT_MIN_BLOCKS_PER_CU > 0
#define WMMA_OPT_LAUNCH_BOUNDS \
    __launch_bounds__(OptConfig::NUM_THREADS, WMMA_OPT_MIN_BLOCKS_PER_CU)
#else
#define WMMA_OPT_LAUNCH_BOUNDS __launch_bounds__(OptConfig::NUM_THREADS)
#endif

// Load A tile: 128 threads load 128 rows × 16 cols
// A is row-major [M, K], stored in LDS as A_lds[M][K] (row-major)
template<typename cfg>
__device__ __forceinline__ void load_A_tile(
    __half A_lds[][cfg::LDS_STRIDE_A],
    const __half* __restrict__ A,
    int block_m, int k_offset, int cid, int M, int K
) {
    const half8_opt zero = {0,0,0,0,0,0,0,0};
    const int a_row = cid;
    if (a_row < cfg::BLOCK_M) {
        // Zero-fill first
        *reinterpret_cast<half8_opt*>(&A_lds[a_row][0]) = zero;
        *reinterpret_cast<half8_opt*>(&A_lds[a_row][8]) = zero;
        
        const int gm_row = block_m + a_row;
        if (gm_row < M) {
            if (k_offset + 8 <= K) {
                *reinterpret_cast<half8_opt*>(&A_lds[a_row][0]) = 
                    *reinterpret_cast<const half8_opt*>(A + gm_row * K + k_offset);
            }
            if (k_offset + 16 <= K) {
                *reinterpret_cast<half8_opt*>(&A_lds[a_row][8]) = 
                    *reinterpret_cast<const half8_opt*>(A + gm_row * K + k_offset + 8);
            }
        }
    }
}

// Load B with vectorized global reads and scalar scatter into transposed LDS.
// Each 8-lane subgroup covers one half8 vector along N for eight K rows. Two
// passes cover BLOCK_K=16. This mirrors the proven 64x16 helper. The former
// shuffle path selected row[lane8] independently in every source lane, copying
// diagonal elements instead of performing an 8x8 transpose.
template<typename cfg>
__device__ __forceinline__ void load_B_tile(
    __half B_lds[][cfg::LDS_STRIDE_B],
    const __half* __restrict__ B,
    int block_n, int k_offset, int cid, int N, int K
) {
    const int lane8 = cid & 7;
    const int n_base = (cid >> 3) * 8;

    #pragma unroll
    for (int k_group = 0; k_group < 2; ++k_group) {
        const int k_base = k_group * 8;
        const int k = k_offset + k_base + lane8;
        const int n_gmem = block_n + n_base;
        half8_opt row = {0,0,0,0,0,0,0,0};
        if (k < K && n_gmem + 7 < N) {
            row = *reinterpret_cast<const half8_opt*>(B + k * N + n_gmem);
        }

        union {
            half8_opt vec;
            _Float16 elems[8];
        } unpacked;
        unpacked.vec = row;

        #pragma unroll
        for (int n_local = 0; n_local < 8; ++n_local) {
            B_lds[n_base + n_local][k_base + lane8] =
                *reinterpret_cast<__half*>(&unpacked.elems[n_local]);
        }
    }
}

template<int CFG_NWARPS = OptConfig::NWARPS>
WMMA_OPT_LAUNCH_BOUNDS
__global__ void wmma_gemm_kernel_opt(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    using cfg = OptConfig;
    
    // A_lds: row-major [BLOCK_M][LDS_STRIDE_A] = [128][24]
    // B_lds: column-major [BLOCK_N][LDS_STRIDE_B] = [128][24]
    __shared__ __half A_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half B_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / cfg::WARP_SIZE;
    const int lane_id = tid % cfg::WARP_SIZE;
    const int half_lane = lane_id % 16;
    const int cid = tid % cfg::HALF_BLOCK;
    const bool load_A = (tid < cfg::HALF_BLOCK);
    
    const int warp_m = warp_id / cfg::WARPS_N;
    const int warp_n = warp_id % cfg::WARPS_N;
    const int warp_m_base = warp_m * cfg::WARP_TILE_M * cfg::WMMA_M;
    const int warp_n_base = warp_n * cfg::WARP_TILE_N * cfg::WMMA_N;
    
    const int block_m = blockIdx.y * cfg::BLOCK_M;
    const int block_n = blockIdx.x * cfg::BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    // Accumulators
    fragment<accumulator, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, float> 
        c_frag[cfg::WARP_TILE_M][cfg::WARP_TILE_N];
    
    #pragma unroll
    for (int i = 0; i < cfg::WARP_TILE_M; i++)
        #pragma unroll
        for (int j = 0; j < cfg::WARP_TILE_N; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // Prologue: load first K-tile
    if (load_A) {
        load_A_tile<cfg>(A_lds[0], A, block_m, 0, cid, M, K);
    } else {
        load_B_tile<cfg>(B_lds[0], B, block_n, 0, cid, N, K);
    }
    __syncthreads();
    
    // Main loop
    int curr_buf = 0;
    
    #pragma unroll 1
    for (int k = 0; k < K; k += cfg::BLOCK_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + cfg::BLOCK_K < K);
        
        // Load fragments from LDS
        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, __half, row_major> 
            a_frag[cfg::WARP_TILE_M];
        fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, __half, col_major> 
            b_frag[cfg::WARP_TILE_N];
        
        #pragma unroll
        for (int ti = 0; ti < cfg::WARP_TILE_M; ti++) {
            load_matrix_sync_lds(a_frag[ti], 
                &A_lds[curr_buf][warp_m_base + ti * cfg::WMMA_M][0], 
                cfg::LDS_STRIDE_A);
        }
        
        #pragma unroll
        for (int tj = 0; tj < cfg::WARP_TILE_N; tj++) {
            // B_lds is [BLOCK_N][LDS_STRIDE_B], column-major
            // For warp_n_base + tj*16, we access B_lds[warp_n_base + tj*16][0]
            load_matrix_sync_lds_b_transposed(b_frag[tj], 
                &B_lds[curr_buf][warp_n_base + tj * cfg::WMMA_N][0], 
                cfg::LDS_STRIDE_B);
        }
        
        // Prefetch next K-tile
        if (has_next) {
            if (load_A) {
                load_A_tile<cfg>(A_lds[next_buf], A, block_m, k + cfg::BLOCK_K, cid, M, K);
            } else {
                load_B_tile<cfg>(B_lds[next_buf], B, block_n, k + cfg::BLOCK_K, cid, N, K);
            }
        }
        
        // Compute
        #pragma unroll
        for (int ti = 0; ti < cfg::WARP_TILE_M; ti++) {
            #pragma unroll
            for (int tj = 0; tj < cfg::WARP_TILE_N; tj++) {
                mma_sync(c_frag[ti][tj], a_frag[ti], b_frag[tj], c_frag[ti][tj]);
            }
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // Epilogue: store fragments to global memory
    const int frag_row_offset = lane_id / 16;
    
    #pragma unroll
    for (int ti = 0; ti < cfg::WARP_TILE_M; ti++) {
        #pragma unroll
        for (int tj = 0; tj < cfg::WARP_TILE_N; tj++) {
            const int tile_row_base = block_m + warp_m_base + ti * cfg::WMMA_M;
            const int tile_col_base = block_n + warp_n_base + tj * cfg::WMMA_N;
            const int c_col = tile_col_base + half_lane;
            
            if (c_col < N) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int c_row = tile_row_base + i * 2 + frag_row_offset;
                    if (c_row < M) {
                        C[c_row * N + c_col] = c_frag[ti][tj].x[i];
                    }
                }
            }
        }
    }
}

template __global__ void wmma_gemm_kernel_opt<OptConfig::NWARPS>(
    const __half*, const __half*, float*, int, int, int);

#undef WMMA_OPT_LAUNCH_BOUNDS

#endif // WMMA_KERNEL_LARGETILE_HPP
