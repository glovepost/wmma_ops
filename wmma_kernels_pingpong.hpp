// ============================================================================
// WMMA PING-PONG KERNEL (V3 - Using Proper Fragment Loading)
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
//
// Uses the same fragment loading pattern as the working kernels
// ============================================================================

#ifndef WMMA_KERNELS_PINGPONG_HPP
#define WMMA_KERNELS_PINGPONG_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

// LDS padding for bank conflict avoidance
#define PINGPONG_LDS_PAD 8

// ============================================================================
// PING-PONG GEMM KERNEL V3
// Uses proper fragment loading via rocwmma helpers
// ============================================================================

template<int NWARPS = 8, int WARPS_M = 4, int WARPS_N = 2>
__launch_bounds__(NWARPS * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void wmma_gemm_kernel_pingpong(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    // ========================================================================
    // CONSTANTS
    // ========================================================================
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = 2 * WMMA_M;  // 32
    constexpr int WARP_TILE_N = 2 * WMMA_N;  // 32
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;  // 128
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;  // 64
    constexpr int BLOCK_K = WMMA_K;  // 16
    
    constexpr int A_STRIDE = BLOCK_K + PINGPONG_LDS_PAD;  // 24
    constexpr int B_STRIDE = BLOCK_K + PINGPONG_LDS_PAD;  // 24
    
    // ========================================================================
    // SHARED MEMORY - Double buffered
    // ========================================================================
    __shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
    
    // ========================================================================
    // THREAD/WARP INDEXING
    // ========================================================================
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Warp position in output tile
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    // Block position
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // ========================================================================
    // ACCUMULATORS (2x2 per warp = 4 WMMA tiles)
    // ========================================================================
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // ========================================================================
    // LOAD INDICES
    // ========================================================================
    const int a_row = tid / 2;
    const int a_col = (tid % 2) * 8;
    const bool a_valid = (a_row < BLOCK_M) && (block_m + a_row < M);
    
    const int b_row = tid / (BLOCK_K / 8);
    const int b_col = (tid % (BLOCK_K / 8)) * 8;
    const bool b_valid = (b_row < BLOCK_N) && (block_n + b_row < N);
    
    // ========================================================================
    // PROLOGUE: Load first K-tile
    // ========================================================================
    if (a_valid && a_col + 8 <= K) {
        half8_t data = *reinterpret_cast<const half8_t*>(A + (block_m + a_row) * K + a_col);
        *reinterpret_cast<half8_t*>(&A_lds[0][a_row][a_col]) = data;
    }
    
    if (b_valid && b_col + 8 <= K) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            if (b_col + k < K) {
                B_lds[0][b_row][b_col + k] = B[(b_col + k) * N + block_n + b_row];
            }
        }
    }
    
    __syncthreads();
    
    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    
    int curr_buf = 0;
    
    #pragma unroll 1
    for (int k = 0; k < K; k += BLOCK_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + BLOCK_K < K);
        
        // Load fragments using rocwmma helpers
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[2];
        
        if (warp_active) {
            #pragma unroll
            for (int ti = 0; ti < 2; ti++)
                load_matrix_sync_lds(a_frag[ti], &A_lds[curr_buf][warp_m_base + ti * WMMA_M][0], A_STRIDE);
            #pragma unroll
            for (int tj = 0; tj < 2; tj++)
                load_matrix_sync_lds_b_transposed(b_frag[tj], &B_lds[curr_buf][warp_n_base + tj * WMMA_N][0], B_STRIDE);
        }
        
        // MMA (0,0) + Prefetch A
        if (warp_active) mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
        
        half8_t a_prefetch = {};
        if (has_next && a_valid && k + BLOCK_K + a_col + 8 <= K) {
            a_prefetch = *reinterpret_cast<const half8_t*>(A + (block_m + a_row) * K + k + BLOCK_K + a_col);
        }
        
        // MMA (0,1) + Prefetch B
        if (warp_active) mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
        
        half8_t b_prefetch = {};
        if (has_next && b_valid && k + BLOCK_K + b_col + 8 <= K) {
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                if (k + BLOCK_K + b_col + kk < K) {
                    reinterpret_cast<__half*>(&b_prefetch)[kk] = 
                        B[(k + BLOCK_K + b_col + kk) * N + block_n + b_row];
                }
            }
        }
        
        // MMA (1,0)
        if (warp_active) mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
        
        // MMA (1,1)
        if (warp_active) mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
        
        // Store prefetched data to LDS
        if (has_next) {
            if (a_valid && k + BLOCK_K + a_col + 8 <= K) {
                *reinterpret_cast<half8_t*>(&A_lds[next_buf][a_row][a_col]) = a_prefetch;
            }
            
            if (b_valid && k + BLOCK_K + b_col + 8 <= K) {
                #pragma unroll
                for (int kk = 0; kk < 8; kk++) {
                    if (k + BLOCK_K + b_col + kk < K) {
                        B_lds[next_buf][b_row][b_col + kk] = reinterpret_cast<__half*>(&b_prefetch)[kk];
                    }
                }
            }
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // ========================================================================
    // EPILOGUE: Store results using standard fragment layout
    // ========================================================================
    if (!warp_active) return;
    
    #pragma unroll
    for (int ti = 0; ti < 2; ti++) {
        #pragma unroll
        for (int tj = 0; tj < 2; tj++) {
            const int tile_row_base = block_m + warp_m_base + ti * WMMA_M;
            const int tile_col_base = block_n + warp_n_base + tj * WMMA_N;
            const int c_col = tile_col_base + (lane_id % 16);
            
            if (c_col < N) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int c_row = tile_row_base + i * 2 + (lane_id / 16);
                    if (c_row < M) C[c_row * N + c_col] = c_frag[ti][tj].x[i];
                }
            }
        }
    }
}

// Explicit template instantiation
template __global__ void wmma_gemm_kernel_pingpong<8, 4, 2>(
    const __half*, const __half*, float*, int, int, int);

#endif // WMMA_KERNELS_PINGPONG_HPP
