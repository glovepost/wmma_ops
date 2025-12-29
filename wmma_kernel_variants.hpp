// ============================================================================
// WMMA OPTIMIZATION VARIANT KERNELS
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
// 
// Alternative kernel implementations with different optimization strategies:
// - K-unrolling: Process 2 K tiles per sync
// - Quad-buffering: Process 4 K tiles per sync
// - High-occupancy: Lower register pressure for higher occupancy
// - No-prefetch: Balanced register pressure
// - Assembly-optimized: Explicit scheduling hints
// ============================================================================

#ifndef WMMA_KERNEL_VARIANTS_HPP
#define WMMA_KERNEL_VARIANTS_HPP

#include "rocwmma_patch/rocwmma_gfx1151.hpp"
#include "wmma_xor_swizzle.hpp"
#include "wmma_device_helpers.hpp"

using namespace rocwmma;

// =============================================================================
// K-UNROLLED KERNEL: Process 2 K tiles per sync
// Reduces __syncthreads overhead from 20% to ~10%
// Expected gain: +2-4 TFLOPS for sync-bound workloads
// =============================================================================

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void wmma_gemm_kernel_kunroll(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int WARP_TILE_M = 2 * WMMA_M;
    constexpr int WARP_TILE_N = 2 * WMMA_N;
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;
    constexpr int BLOCK_K = WMMA_K;
    constexpr int K_UNROLL = 2;
    constexpr int SUPER_K = BLOCK_K * K_UNROLL;  // 32
    
    // LDS stride with padding for bank conflict reduction
    constexpr int A_STRIDE = SUPER_K + LDS_PAD;  // 40
    constexpr int B_STRIDE = SUPER_K + LDS_PAD;  // 40
    
    __shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // Thread mapping for A: 128 rows × 32 cols per SUPER_K
    // Each thread loads one half8 per K-tile (2 half8 total per SUPER_K)
    constexpr int A_VECS_PER_ROW = BLOCK_K / 8;  // 2 vectors per row per K-tile
    constexpr int A_VEC_LOADS = BLOCK_M * A_VECS_PER_ROW;  // 256 threads needed per K-tile
    const int a_row = tid / A_VECS_PER_ROW;
    const int a_vec_in_row = tid % A_VECS_PER_ROW;  // 0 or 1 (which half8 in the row)
    const bool a_valid = (tid < A_VEC_LOADS) && (block_m + a_row < M);
    
    // Thread mapping for B: 16 rows × 64 cols per K-tile, transposed to (N,K)
    constexpr int B_VECS_PER_K = BLOCK_N / 8;  // 8 vectors per K row
    constexpr int B_VEC_LOADS = BLOCK_K * B_VECS_PER_K;  // 128 threads needed per K-tile
    const int b_k_in_tile = tid / B_VECS_PER_K;  // 0..15 (row within a K-tile)
    const int b_n = (tid % B_VECS_PER_K) * 8;  // Column in B
    const bool b_valid = (tid < B_VEC_LOADS) && (block_n + b_n + 7 < N);
    
    const __half* A_ptr = A + (block_m + a_row) * K;
    const __half* B_ptr = B + block_n + b_n;
    
    // Helper lambda to load one SUPER_K chunk into specified LDS buffer
    // k_start: global K offset to load from
    auto load_super_k = [&](int buf, int k_start) {
        #pragma unroll
        for (int ku = 0; ku < K_UNROLL; ku++) {
            const int k_local = ku * BLOCK_K;  // Offset within LDS SUPER_K region
            const int k_global = k_start + k_local;  // Global K coordinate
            
            // Load A: each thread loads one half8 per K-tile
            const int a_col_local = a_vec_in_row * 8 + k_local;  // LDS column
            const int a_col_global = a_vec_in_row * 8 + k_global;  // Global K + column offset
            if (a_valid && a_col_global + 8 <= K) {
                *reinterpret_cast<half8*>(&A_lds[buf][a_row][a_col_local]) = 
                    *reinterpret_cast<const half8*>(A_ptr + a_col_global);
            }
            
            // B: vectorized transpose via shuffle for this K-tile
            // Note: B_STRIDE here is SUPER_K + LDS_PAD = 40, and we write at offset k_local
            load_B_tile_vec_64x16(&B_lds[buf][0][k_local], B, N, K, block_n, k_global, tid, B_STRIDE);
        }
    };
    
    // PROLOGUE: Load first SUPER_K chunk
    load_super_k(0, 0);
    __syncthreads();
    
    int curr_buf = 0;
    
    // MAIN LOOP: SUPER_K per iteration, only 1 sync per 2 K tiles
    #pragma unroll 1
    for (int k = 0; k < K; k += SUPER_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + SUPER_K < K);
        
        // Process K_UNROLL tiles from current buffer (NO sync between them)
        #pragma unroll
        for (int ku = 0; ku < K_UNROLL; ku++) {
            const int k_global = k + ku * BLOCK_K;
            const int k_local = ku * BLOCK_K;
            
            if (k_global >= K) break;
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[2];
            
            if (warp_active) {
                #pragma unroll
                for (int ti = 0; ti < 2; ti++)
                    load_matrix_sync_lds(a_frag[ti], 
                        &A_lds[curr_buf][warp_m_base + ti*WMMA_M][k_local], A_STRIDE);
                #pragma unroll
                for (int tj = 0; tj < 2; tj++)
                    load_matrix_sync_lds_b_transposed(b_frag[tj],
                        &B_lds[curr_buf][warp_n_base + tj*WMMA_N][k_local], B_STRIDE);
                
                #pragma unroll
                for (int ti = 0; ti < 2; ti++)
                    #pragma unroll
                    for (int tj = 0; tj < 2; tj++)
                        mma_sync(c_frag[ti][tj], a_frag[ti], b_frag[tj], c_frag[ti][tj]);
            }
        }
        
        // Prefetch next SUPER_K chunk
        if (has_next) {
            load_super_k(next_buf, k + SUPER_K);
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // EPILOGUE
    if (warp_active) {
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
}

template __global__ void wmma_gemm_kernel_kunroll<8, 4, 2>(const __half*, const __half*, float*, int, int, int);

// =============================================================================
// Experimental: Quad-buffered kernel (sync every 64 K elements)
// Goal: Reduce __syncthreads overhead by 4x
// =============================================================================

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 2)
__global__ void wmma_gemm_kernel_quad(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int WARP_TILE_M = 2 * WMMA_M;
    constexpr int WARP_TILE_N = 2 * WMMA_N;
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;
    constexpr int BLOCK_K = WMMA_K;  // 16
    constexpr int UNROLL_K = 4;       // Process 4 K tiles per sync
    constexpr int SUPER_K = BLOCK_K * UNROLL_K;  // 64
    
    constexpr int A_STRIDE = BLOCK_K;
    constexpr int B_STRIDE = BLOCK_K;
    
    // 4 buffers for quad-buffering (double the LDS but 4x fewer syncs)
    __shared__ __half A_lds[UNROLL_K][BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[UNROLL_K][BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
            #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // Load assignments for A
    constexpr int A_VECS_PER_ROW = BLOCK_K / 8;  // 2
    constexpr int A_VEC_LOADS = BLOCK_M * A_VECS_PER_ROW;  // 256
    const int a_row = tid / A_VECS_PER_ROW;
    const int a_col = (tid % A_VECS_PER_ROW) * 8;
    const bool a_valid = (tid < A_VEC_LOADS) && (block_m + a_row < M);
    
    // Load assignments for B (transposed into LDS)
    constexpr int B_VECS_PER_K = BLOCK_N / 8;  // 8
    constexpr int B_VEC_LOADS = BLOCK_K * B_VECS_PER_K;  // 128
    const int b_k = tid / B_VECS_PER_K;
    const int b_n = (tid % B_VECS_PER_K) * 8;
    const bool b_valid = (tid < B_VEC_LOADS) && (block_n + b_n + 7 < N);
    
    // MAIN LOOP (64 K elements per iteration)
    for (int k_super = 0; k_super < K; k_super += SUPER_K) {
        // Load 4 tiles into LDS buffers
        #pragma unroll
        for (int buf = 0; buf < UNROLL_K; buf++) {
            const int k = k_super + buf * BLOCK_K;
            
            // Load A
            if (a_valid && k + a_col + 8 <= K) {
                *reinterpret_cast<half8*>(&A_lds[buf][a_row][a_col]) = 
                    *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + k + a_col);
            }
            
            // B: vectorized transpose via shuffle
            load_B_tile_vec_64x16(&B_lds[buf][0][0], B, N, K, block_n, k, tid, B_STRIDE);
        }
        
        __syncthreads();
        
        // Compute tiles using LDS buffers
        // FIX: Only compute with buffers that have valid K data
        #pragma unroll
        for (int buf = 0; buf < UNROLL_K; buf++) {
            const int k = k_super + buf * BLOCK_K;
            // Skip this buffer if it's beyond K boundary
            if (k >= K) break;
            
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[2];
            
            if (warp_active) {
            #pragma unroll
                for (int ti = 0; ti < 2; ti++)
                    load_matrix_sync_lds(a_frag[ti], &A_lds[buf][warp_m_base + ti * WMMA_M][0], A_STRIDE);
                #pragma unroll
                for (int tj = 0; tj < 2; tj++)
                    load_matrix_sync_lds_b_transposed(b_frag[tj], &B_lds[buf][warp_n_base + tj * WMMA_N][0], B_STRIDE);
                
            #pragma unroll
                for (int ti = 0; ti < 2; ti++)
                    #pragma unroll
                    for (int tj = 0; tj < 2; tj++)
                        mma_sync(c_frag[ti][tj], a_frag[ti], b_frag[tj], c_frag[ti][tj]);
            }
        }
        
        __syncthreads();
    }
    
    // EPILOGUE
    if (warp_active) {
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
}

template __global__ void wmma_gemm_kernel_quad<8, 4, 2>(const __half*, const __half*, float*, int, int, int);

// =============================================================================
// HIGH OCCUPANCY KERNEL: Target 64 VGPRs -> 8 waves/CU
//
// Optimizations for lower register pressure:
// 1. 2x1 register blocking (2 accumulators instead of 4)
// 2. No prefetch registers (direct GMEM -> LDS)
// 3. Fewer warps per block (4 instead of 8)
// 4. Explicit __launch_bounds__ with VGPR hint
// =============================================================================

// Use __attribute__ to hint max VGPRs (HIP extension)
template<int NWARPS>
__launch_bounds__(NWARPS * 32, 4)  // Target 4 blocks/CU for higher occupancy
__global__ void wmma_gemm_kernel_highOcc(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    // Simpler layout: 2x2 warps, but 2x1 register blocking per warp
    constexpr int WARPS_M = 2;
    constexpr int WARPS_N = 2;
    constexpr int WARP_TILE_M = 2 * WMMA_M;  // 32 rows per warp (2 tiles)
    constexpr int WARP_TILE_N = 1 * WMMA_N;  // 16 cols per warp (1 tile) <- reduced!
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;  // 64
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;  // 32
    constexpr int BLOCK_K = WMMA_K;                  // 16
    
    constexpr int A_STRIDE = BLOCK_K;
    constexpr int B_STRIDE = BLOCK_K;
    
    // Single buffer LDS (no double buffering to save registers)
    __shared__ __half A_lds[BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // Only 2 accumulators (2x1 blocking) - saves 16 VGPRs vs 4 accumulators
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2];
    fill_fragment(c_frag[0], 0.0f);
    fill_fragment(c_frag[1], 0.0f);
    
    // Load indices - simplified, computed once
    constexpr int A_ELEMS = BLOCK_M * BLOCK_K;  // 64*16 = 1024
    constexpr int B_ELEMS = BLOCK_K * BLOCK_N;  // 16*32 = 512
 // 128
    
    // Main K loop
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A: 1024 elements / 128 threads = 8 elements per thread
        // Each thread loads 8 consecutive halfs (one half8)
        const int a_idx = tid * 8;
        if (a_idx < A_ELEMS) {
            const int a_row = a_idx / BLOCK_K;
            const int a_col = a_idx % BLOCK_K;
            if (block_m + a_row < M && k + a_col + 8 <= K) {
                *reinterpret_cast<half8*>(&A_lds[a_row][a_col]) = 
                    *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + k + a_col);
            }
        }
        
        // Load B transposed: 512 elements / 128 threads = 4 elements per thread
        // Use scalar loads for transpose
        for (int i = 0; i < 4; i++) {
            const int b_idx = tid * 4 + i;
            if (b_idx < B_ELEMS) {
                const int b_k = b_idx / BLOCK_N;
                const int b_n = b_idx % BLOCK_N;
                if (k + b_k < K && block_n + b_n < N) {
                    B_lds[b_n][b_k] = B[(k + b_k) * N + block_n + b_n];
                }
            }
        }
        
        __syncthreads();
    
        // Compute: 2x1 tiles per warp
        if (warp_active) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
            
            // Load A tiles (2 tiles in M direction)
            load_matrix_sync_lds(a_frag[0], &A_lds[warp_m_base][0], A_STRIDE);
            load_matrix_sync_lds(a_frag[1], &A_lds[warp_m_base + WMMA_M][0], A_STRIDE);
            
            // Load B tile (1 tile in N direction)
            load_matrix_sync_lds_b_transposed(b_frag, &B_lds[warp_n_base][0], B_STRIDE);
            
            // 2 MMAs (vs 4 in the original)
            mma_sync(c_frag[0], a_frag[0], b_frag, c_frag[0]);
            mma_sync(c_frag[1], a_frag[1], b_frag, c_frag[1]);
        }
        
        __syncthreads();
    }
    
    // Store results
    if (warp_active) {
        for (int ti = 0; ti < 2; ti++) {
            const int tile_row_base = block_m + warp_m_base + ti * WMMA_M;
            const int tile_col_base = block_n + warp_n_base;
            const int c_col = tile_col_base + (lane_id % 16);
            
    if (c_col < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
                    const int c_row = tile_row_base + i * 2 + (lane_id / 16);
                    if (c_row < M) C[c_row * N + c_col] = c_frag[ti].x[i];
                }
            }
        }
    }
}

template __global__ void wmma_gemm_kernel_highOcc<4>(const __half*, const __half*, float*, int, int, int);

// =============================================================================
// BALANCED OCCUPANCY: Keep 2x2 blocking but remove prefetch registers
// Target: ~75 VGPRs -> 6 waves/CU (vs 5 currently)
// =============================================================================

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 3)  // Target 3 blocks for ~6 waves
__attribute__((amdgpu_waves_per_eu(4, 8)))  // Hint: prefer 4-8 waves per EU
__global__ void wmma_gemm_kernel_noPrefetch(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int WARP_TILE_M = 2 * WMMA_M;
    constexpr int WARP_TILE_N = 2 * WMMA_N;
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;
    constexpr int BLOCK_K = WMMA_K;
    
    // [Optimization] LDS Padding to eliminate bank conflicts
    constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 16 + 8 = 24
    constexpr int B_STRIDE = BLOCK_K + LDS_PAD;  // 16 + 8 = 24
    
    constexpr int A_ELEMS = BLOCK_M * BLOCK_K;
    constexpr int B_ELEMS = BLOCK_K * BLOCK_N;
    constexpr int A_VEC_LOADS = A_ELEMS / 8;
    constexpr int B_VEC_LOADS = B_ELEMS / 8;
    
    // Single buffer (no double buffering to reduce complexity)
    __shared__ __half A_lds[BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // 4 accumulators (2x2 blocking)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // Load indices
    constexpr int A_VECS_PER_ROW = BLOCK_K / 8;
    const int a_row = tid / A_VECS_PER_ROW;
    const int a_col = (tid % A_VECS_PER_ROW) * 8;
    const bool a_valid = (tid < A_VEC_LOADS) && (block_m + a_row < M);
    
    constexpr int B_VECS_PER_K = BLOCK_N / 8;
    const int b_k = tid / B_VECS_PER_K;
    const int b_n = (tid % B_VECS_PER_K) * 8;
    const bool b_valid = (tid < B_VEC_LOADS) && (b_k < BLOCK_K) && (block_n + b_n + 7 < N);
    
    // Main loop - NO prefetch registers, direct load each iteration
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load A directly (no prefetch register)
        if (a_valid && k + a_col + 8 <= K) {
            *reinterpret_cast<half8*>(&A_lds[a_row][a_col]) = 
                *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + k + a_col);
        }
        
        // B: vectorized transpose via shuffle
        load_B_tile_vec_64x16(&B_lds[0][0], B, N, K, block_n, k, tid, B_STRIDE);
        
        __syncthreads();
        
        // Compute 2x2 tiles
        if (warp_active) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[2];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[2];
            
            #pragma unroll
            for (int ti = 0; ti < 2; ti++)
                load_matrix_sync_lds(a_frag[ti], &A_lds[warp_m_base + ti * WMMA_M][0], A_STRIDE);
            #pragma unroll
            for (int tj = 0; tj < 2; tj++)
                load_matrix_sync_lds_b_transposed(b_frag[tj], &B_lds[warp_n_base + tj * WMMA_N][0], B_STRIDE);
            
            #pragma unroll
            for (int ti = 0; ti < 2; ti++)
                #pragma unroll
                for (int tj = 0; tj < 2; tj++)
                    mma_sync(c_frag[ti][tj], a_frag[ti], b_frag[tj], c_frag[ti][tj]);
        }
        
        __syncthreads();
    }
    
    // Store
    if (warp_active) {
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
}

template __global__ void wmma_gemm_kernel_noPrefetch<8, 4, 2>(const __half*, const __half*, float*, int, int, int);

// =============================================================================
// ASSEMBLY-OPTIMIZED KERNEL: Explicit scheduling hints
// Uses __builtin_expect and manual unrolling for better instruction flow
// =============================================================================

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 2)
__global__ void wmma_gemm_kernel_asmOpt(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int WARP_TILE_M = 2 * WMMA_M;
    constexpr int WARP_TILE_N = 2 * WMMA_N;
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;
    constexpr int BLOCK_K = WMMA_K;
    
    // LDS padding for bank conflict avoidance (consistent with other kernels)
    constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 24
    constexpr int B_STRIDE = BLOCK_K + LDS_PAD;  // 24
    
    // Use register keyword hints for critical variables
    __shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (__builtin_expect(block_m >= M || block_n >= N, 0)) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // Accumulators - use explicit float8_t for better register allocation
    // float8_t is defined in wmma_xor_swizzle.hpp
    float8_t c00 = {}, c01 = {}, c10 = {}, c11 = {};
    
    // Precompute load indices (reused every iteration)
    constexpr int A_VECS_PER_ROW = BLOCK_K / 8;
    const int a_row = tid / A_VECS_PER_ROW;
    const int a_col = (tid % A_VECS_PER_ROW) * 8;
    const bool a_valid = (tid < (BLOCK_M * A_VECS_PER_ROW)) && (block_m + a_row < M);
    
    // Precompute base pointer for A
    const __half* A_base = A + (block_m + a_row) * K;
    
    // PROLOGUE: First tile load
    if (a_valid && a_col + 8 <= K) {
        *reinterpret_cast<half8*>(&A_lds[0][a_row][a_col]) = 
            *reinterpret_cast<const half8*>(A_base + a_col);
    }
    
    // B: vectorized transpose via shuffle
    load_B_tile_vec_64x16(&B_lds[0][0][0], B, N, K, block_n, 0, tid, B_STRIDE);
    
    __syncthreads();
    
    int curr_buf = 0;
    
    // Fragment row/col index for this lane
    // For RDNA3 WMMA, lanes 0-15 and 16-31 must have replicated data
    // AMD GPUOpen pattern:
    //   A fragment: each lane loads its own ROW of A (all 16 K values)
    //   B fragment: each lane loads its own COLUMN of B (all 16 K values)
    const int frag_idx = lane_id % 16;
    
    // MAIN LOOP
    #pragma unroll 1  // Prevent over-unrolling which increases register pressure
    for (int k = 0; k < K; k += BLOCK_K) {
        const int next_k = k + BLOCK_K;
        const int next_buf = 1 - curr_buf;
        const bool has_next = __builtin_expect(next_k < K, 1);
        
        half8 a_prefetch = {};
        half8 b_prefetch = {};
        
        // FIXED: Load fragments from LDS using correct AMD GPUOpen pattern
        // A fragment: each lane loads ROW frag_idx (all 16 K values)
        // B fragment: each lane loads ROW frag_idx from transposed B_lds[N][K] 
        //             (which is COLUMN frag_idx of original B)
        half16_t a0, a1, b0, b1;
        
        // A: Load row frag_idx, all K values (vectorized as 2x half8)
        const __half* a0_row = &A_lds[curr_buf][warp_m_base + frag_idx][0];
        const __half* a1_row = &A_lds[curr_buf][warp_m_base + 16 + frag_idx][0];
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            a0[kk] = *reinterpret_cast<const _Float16*>(&a0_row[kk]);
            a0[kk + 8] = *reinterpret_cast<const _Float16*>(&a0_row[kk + 8]);
            a1[kk] = *reinterpret_cast<const _Float16*>(&a1_row[kk]);
            a1[kk + 8] = *reinterpret_cast<const _Float16*>(&a1_row[kk + 8]);
        }
        
        // B: Load row frag_idx from transposed B_lds (= column frag_idx of original B)
        const __half* b0_row = &B_lds[curr_buf][warp_n_base + frag_idx][0];
        const __half* b1_row = &B_lds[curr_buf][warp_n_base + 16 + frag_idx][0];
        #pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            b0[kk] = *reinterpret_cast<const _Float16*>(&b0_row[kk]);
            b0[kk + 8] = *reinterpret_cast<const _Float16*>(&b0_row[kk + 8]);
            b1[kk] = *reinterpret_cast<const _Float16*>(&b1_row[kk]);
            b1[kk + 8] = *reinterpret_cast<const _Float16*>(&b1_row[kk + 8]);
        }
        
        // ============ COMPUTE + PREFETCH ============
        // Issue first 2 WMMAs + global loads (dual-issue opportunity)
        c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
        
        // Prefetch A for next iteration (should dual-issue with WMMA pipeline)
        if (has_next && a_valid && next_k + a_col + 8 <= K)
            a_prefetch = *reinterpret_cast<const half8*>(A_base + next_k + a_col);
        
        c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
        
        // Issue remaining 2 WMMAs
        c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
        c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        
        // Write prefetch to LDS for next iteration (cooperative vectorized loads)
        if (has_next) {
            if (a_valid && next_k + a_col + 8 <= K)
                *reinterpret_cast<half8*>(&A_lds[next_buf][a_row][a_col]) = a_prefetch;
            
            load_B_tile_vec_64x16(&B_lds[next_buf][0][0], B, N, K, block_n, next_k, tid, B_STRIDE);
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // EPILOGUE: Store results
    if (warp_active) {
        const int base_col0 = block_n + warp_n_base + (lane_id % 16);
        const int base_col1 = base_col0 + 16;
        
    #pragma unroll
        for (int ti = 0; ti < 2; ti++) {
            const int tile_row_base = block_m + warp_m_base + ti * WMMA_M;
            const float8_t& c_tile0 = (ti == 0) ? c00 : c10;
            const float8_t& c_tile1 = (ti == 0) ? c01 : c11;
            
            if (base_col0 < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
                    const int c_row = tile_row_base + i * 2 + (lane_id / 16);
                    if (c_row < M) C[c_row * N + base_col0] = c_tile0[i];
                }
            }
            if (base_col1 < N) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int c_row = tile_row_base + i * 2 + (lane_id / 16);
                    if (c_row < M) C[c_row * N + base_col1] = c_tile1[i];
                }
            }
        }
    }
}

template __global__ void wmma_gemm_kernel_asmOpt<8, 4, 2>(const __half*, const __half*, float*, int, int, int);

// =============================================================================
// HILBERT-OPTIMIZED KERNEL V2
//
// Applies optimizations from AMD's perf_hgemm.cpp:
// 1. 4×2 warp tiles (64×32 per warp, 8 WMMA tiles = 8 accumulators)
// 2. Double-buffered LDS with pointer swap
// 3. Prefetch pipeline (global→LDS during MMA)
// 4. Hilbert curve tile mapping for L2 cache locality
// =============================================================================

#include "wmma_tile_mapping.hpp"

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void wmma_gemm_kernel_hilbert(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    const int grid_m, const int grid_n
) {
    // ========================================================================
    // TILE CONFIGURATION: 8 warps, 2×2 tiles per warp = 256 threads
    // This matches thread count to load requirements (256 A vectors = 256 threads)
    // ========================================================================
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    // Each warp computes 2×2 = 4 WMMA tiles (32×32 output)
    constexpr int BLOCKS_M = 2;  // 2 tiles in M direction per warp
    constexpr int BLOCKS_N = 2;  // 2 tiles in N direction per warp
    constexpr int WARP_TILE_M = BLOCKS_M * WMMA_M;  // 32
    constexpr int WARP_TILE_N = BLOCKS_N * WMMA_N;  // 32
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int MACRO_TILE_M = WARPS_M * WARP_TILE_M;  // 128 (with 4×2 warps)
    constexpr int MACRO_TILE_N = WARPS_N * WARP_TILE_N;  // 64  (with 4×2 warps)
    constexpr int MACRO_TILE_K = WMMA_K;
    
    constexpr int A_STRIDE = MACRO_TILE_K + LDS_PAD;
    constexpr int B_STRIDE = MACRO_TILE_K + LDS_PAD;
    constexpr int NTHREADS = NWARPS * 32;
    
    // Double-buffered LDS (2 buffers for A+B each)
    constexpr int LDS_A_SIZE = MACRO_TILE_M * A_STRIDE;
    constexpr int LDS_B_SIZE = MACRO_TILE_N * B_STRIDE;
    __shared__ __half lds_buffer[2 * (LDS_A_SIZE + LDS_B_SIZE)];
    
    // LDS buffer pointers (will be swapped)
    __half* A_lds0 = &lds_buffer[0];
    __half* B_lds0 = &lds_buffer[LDS_A_SIZE];
    __half* A_lds1 = &lds_buffer[LDS_A_SIZE + LDS_B_SIZE];
    __half* B_lds1 = &lds_buffer[2 * LDS_A_SIZE + LDS_B_SIZE];
    
    __half* A_lds_curr = A_lds0;
    __half* B_lds_curr = B_lds0;
    __half* A_lds_next = A_lds1;
    __half* B_lds_next = B_lds1;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    // ========================================================================
    // HILBERT TILE MAPPING (1D grid launch)
    // ========================================================================
    int block_row, block_col;
    tile_mapping::hilbert_tile_mapping<MACRO_TILE_M, MACRO_TILE_N>(
        blockIdx.x, grid_m, grid_n, &block_row, &block_col);
    
    const int block_m = block_row;
    const int block_n = block_col;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // ========================================================================
    // ACCUMULATORS: 2×2 = 4 fragments per warp
    // ========================================================================
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[BLOCKS_M][BLOCKS_N];
    #pragma unroll
    for (int i = 0; i < BLOCKS_M; i++)
        #pragma unroll
        for (int j = 0; j < BLOCKS_N; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // ========================================================================
    // COOPERATIVE LOADING: Single-vector per thread (256 threads = 256 vectors for A)
    // A: MACRO_TILE_M × MACRO_TILE_K = 128×16 = 2048 elements = 256 vectors
    // B: MACRO_TILE_K × MACRO_TILE_N = 16×64 = 1024 elements = 128 vectors
    // ========================================================================
    constexpr int A_VEC_LOADS = (MACRO_TILE_M * MACRO_TILE_K) / 8;  // 256
    constexpr int B_VEC_LOADS = (MACRO_TILE_K * MACRO_TILE_N) / 8;  // 128
    
    // A loading indices (256 threads, 256 vectors = 1 per thread)
    constexpr int A_VECS_PER_ROW = MACRO_TILE_K / 8;  // 2
    const int a_row = tid / A_VECS_PER_ROW;
    const int a_col = (tid % A_VECS_PER_ROW) * 8;
    const bool a_valid = (tid < A_VEC_LOADS) && (block_m + a_row < M);
    
    // B loading indices (128 threads load B, tid < 128)
    constexpr int B_VECS_PER_K = MACRO_TILE_N / 8;  // 8
    const int b_k = tid / B_VECS_PER_K;
    const int b_n = (tid % B_VECS_PER_K) * 8;
    const bool b_valid = (tid < B_VEC_LOADS) && (b_k < MACRO_TILE_K) && (block_n + b_n + 7 < N);
    
    // ========================================================================
    // PROLOGUE: Load first K-tile to buffer 0
    // ========================================================================
    // A: single vector per thread (256 threads = 256 vectors)
    if (a_valid && a_col + 8 <= K) {
        *reinterpret_cast<half8*>(&A_lds_curr[a_row * A_STRIDE + a_col]) = 
            *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + a_col);
    }
    
    // B: vectorized transpose via shuffle
    load_B_tile_vec_64x16(B_lds_curr, B, N, K, block_n, 0, tid, B_STRIDE);
    
    __syncthreads();
    
    // ========================================================================
    // MAIN LOOP: Double-buffered prefetch (perf_hgemm pattern)
    // ========================================================================
    for (int k = MACRO_TILE_K; k < K; k += MACRO_TILE_K) {
        // Step 1: Prefetch NEXT K-tile to alternate buffer (global → LDS)
        if (a_valid && k + a_col + 8 <= K) {
            *reinterpret_cast<half8*>(&A_lds_next[a_row * A_STRIDE + a_col]) = 
                *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + k + a_col);
        }
        
        // B: vectorized transpose via shuffle
        load_B_tile_vec_64x16(B_lds_next, B, N, K, block_n, k, tid, B_STRIDE);
        
        // Step 2: Load fragments from CURRENT buffer and compute MMA
        if (warp_active) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[BLOCKS_M];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[BLOCKS_N];
            
            #pragma unroll
            for (int ti = 0; ti < BLOCKS_M; ti++) {
                load_matrix_sync_lds(a_frag[ti], 
                    &A_lds_curr[(warp_m_base + ti * WMMA_M) * A_STRIDE], A_STRIDE);
            }
            #pragma unroll
            for (int tj = 0; tj < BLOCKS_N; tj++) {
                load_matrix_sync_lds_b_transposed(b_frag[tj], 
                    &B_lds_curr[(warp_n_base + tj * WMMA_N) * B_STRIDE], B_STRIDE);
            }
            
            // 4×2 = 8 MMAs per warp
            #pragma unroll
            for (int ti = 0; ti < BLOCKS_M; ti++)
                #pragma unroll
                for (int tj = 0; tj < BLOCKS_N; tj++)
                    mma_sync(c_frag[ti][tj], a_frag[ti], b_frag[tj], c_frag[ti][tj]);
        }
        
        // Step 3: Sync and swap buffers
        __syncthreads();
        
        __half* tmp = A_lds_curr;
        A_lds_curr = A_lds_next;
        A_lds_next = tmp;
        
        tmp = B_lds_curr;
        B_lds_curr = B_lds_next;
        B_lds_next = tmp;
    }
    
    // ========================================================================
    // TAIL: Process last K-tile (no prefetch needed)
    // ========================================================================
    if (warp_active) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag[BLOCKS_M];
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag[BLOCKS_N];
        
        #pragma unroll
        for (int ti = 0; ti < BLOCKS_M; ti++) {
            load_matrix_sync_lds(a_frag[ti], 
                &A_lds_curr[(warp_m_base + ti * WMMA_M) * A_STRIDE], A_STRIDE);
        }
        #pragma unroll
        for (int tj = 0; tj < BLOCKS_N; tj++) {
            load_matrix_sync_lds_b_transposed(b_frag[tj], 
                &B_lds_curr[(warp_n_base + tj * WMMA_N) * B_STRIDE], B_STRIDE);
        }
        
        #pragma unroll
        for (int ti = 0; ti < BLOCKS_M; ti++)
            #pragma unroll
            for (int tj = 0; tj < BLOCKS_N; tj++)
                mma_sync(c_frag[ti][tj], a_frag[ti], b_frag[tj], c_frag[ti][tj]);
    }
    
    // ========================================================================
    // EPILOGUE: Store 4×2 tiles
    // ========================================================================
    if (warp_active) {
        #pragma unroll
        for (int ti = 0; ti < BLOCKS_M; ti++) {
            #pragma unroll
            for (int tj = 0; tj < BLOCKS_N; tj++) {
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
}

// Template instantiation for 4 warps in M, 2 warps in N = 8 warps total = 256 threads
// Each warp: 32×32. Macro tile: 128×64
template __global__ void wmma_gemm_kernel_hilbert<8, 4, 2>(
    const __half*, const __half*, float*, int, int, int, int, int);


// =============================================================================
// COOPERATIVE LOADING KERNEL
// 
// Optimization: Split threads into two halves - first half loads A, second half
// loads B simultaneously. This maximizes parallelism during the LDS loading phase.
//
// Benefits:
// - A and B loaded in parallel (not sequentially)
// - Better thread utilization during load phase
// - Reduced effective load latency
// =============================================================================

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void wmma_gemm_kernel_coop(
    const __half* __restrict__ A, 
    const __half* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int WARP_TILE_M = 2 * WMMA_M;  // 32
    constexpr int WARP_TILE_N = 2 * WMMA_N;  // 32
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;  // 128 for 4 warps
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;  // 64 for 2 warps
    constexpr int BLOCK_K = WMMA_K;                  // 16
    
    constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 24
    constexpr int B_STRIDE = BLOCK_K + LDS_PAD;  // 24
    
    __shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // Accumulators
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            fill_fragment(c_frag[i][j], 0.0f);
    
    // ========================================================================
    // COOPERATIVE LOADING: Split threads into A-loaders and B-loaders
    // 256 threads total: 128 load A, 128 load B
    // ========================================================================
    constexpr int NUM_THREADS = NWARPS * WARP_SIZE;  // 256
    constexpr int HALF_THREADS = NUM_THREADS / 2;     // 128
    
    const bool is_a_loader = (tid < HALF_THREADS);
    const int coop_tid = tid % HALF_THREADS;  // Local thread ID within A or B group
    
    // A loading: 128 threads load 128×16 = 2048 halfs = 256 half8 vectors
    // Each of 128 threads loads 2 half8 vectors
    constexpr int A_VECS_TOTAL = BLOCK_M * BLOCK_K / 8;  // 256
    constexpr int A_VECS_PER_THREAD = A_VECS_TOTAL / HALF_THREADS;  // 2
    
    // B loading: 128 threads load 16×64 = 1024 halfs = 128 half8 vectors
    // Each of 128 threads loads 1 half8 vector
    constexpr int B_VECS_TOTAL = BLOCK_K * BLOCK_N / 8;  // 128
    constexpr int B_VECS_PER_THREAD = B_VECS_TOTAL / HALF_THREADS;  // 1
    
    // Pre-compute load indices
    int a_row[A_VECS_PER_THREAD], a_col[A_VECS_PER_THREAD];
    bool a_valid[A_VECS_PER_THREAD];
    
    if (is_a_loader) {
        #pragma unroll
        for (int v = 0; v < A_VECS_PER_THREAD; v++) {
            const int vec_idx = coop_tid * A_VECS_PER_THREAD + v;
            a_row[v] = vec_idx / (BLOCK_K / 8);
            a_col[v] = (vec_idx % (BLOCK_K / 8)) * 8;
            a_valid[v] = (block_m + a_row[v] < M);
        }
    }
    
    int b_k, b_n;
    bool b_valid_flag;
    
    if (!is_a_loader) {
        const int vec_idx = coop_tid;
        b_k = vec_idx / (BLOCK_N / 8);
        b_n = (vec_idx % (BLOCK_N / 8)) * 8;
        b_valid_flag = (b_k < BLOCK_K) && (block_n + b_n + 7 < N);
    }
    
    // Base pointers
    const __half* A_base = A + block_m * K;
    const __half* B_base = B + block_n;
    
    // ========================================================================
    // PROLOGUE: Cooperative load of first tile
    // ========================================================================
    if (is_a_loader) {
        #pragma unroll
        for (int v = 0; v < A_VECS_PER_THREAD; v++) {
            if (a_valid[v] && a_col[v] + 8 <= K) {
                *reinterpret_cast<half8*>(&A_lds[0][a_row[v]][a_col[v]]) = 
                    *reinterpret_cast<const half8*>(A_base + a_row[v] * K + a_col[v]);
            }
        }
    } else {
        // B: vectorized transpose via shuffle (use coop_tid which is 0-127 for B loaders)
        load_B_tile_vec_64x16(&B_lds[0][0][0], B, N, K, block_n, 0, coop_tid, B_STRIDE);
    }
    
    __syncthreads();
    
    int curr_buf = 0;
    
    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    for (int k = 0; k < K; k += BLOCK_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + BLOCK_K < K);
        
        // Load fragments and compute
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
        
        // Prefetch next tile cooperatively
        half8 a_prefetch[A_VECS_PER_THREAD] = {};
        half8 b_prefetch = {};
        
        if (has_next) {
            if (is_a_loader) {
                #pragma unroll
                for (int v = 0; v < A_VECS_PER_THREAD; v++) {
                    if (a_valid[v] && (k + BLOCK_K + a_col[v] + 8 <= K)) {
                        a_prefetch[v] = *reinterpret_cast<const half8*>(
                            A_base + a_row[v] * K + (k + BLOCK_K) + a_col[v]);
                    }
                }
            } else {
                if (b_valid_flag && (k + BLOCK_K + b_k < K)) {
                    b_prefetch = *reinterpret_cast<const half8*>(
                        B_base + (k + BLOCK_K + b_k) * N + b_n);
                }
            }
        }
        
        // MMA compute
        if (warp_active) {
            mma_sync(c_frag[0][0], a_frag[0], b_frag[0], c_frag[0][0]);
            mma_sync(c_frag[0][1], a_frag[0], b_frag[1], c_frag[0][1]);
            mma_sync(c_frag[1][0], a_frag[1], b_frag[0], c_frag[1][0]);
            mma_sync(c_frag[1][1], a_frag[1], b_frag[1], c_frag[1][1]);
        }
        
        // Write prefetch to LDS
        if (has_next) {
            if (is_a_loader) {
                #pragma unroll
                for (int v = 0; v < A_VECS_PER_THREAD; v++) {
                    if (a_valid[v]) {
                        *reinterpret_cast<half8*>(&A_lds[next_buf][a_row[v]][a_col[v]]) = a_prefetch[v];
                    }
                }
            } else {
                // B: vectorized transpose via shuffle
                load_B_tile_vec_64x16(&B_lds[next_buf][0][0], B, N, K, block_n, k + BLOCK_K, coop_tid, B_STRIDE);
            }
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // ========================================================================
    // EPILOGUE: Store results
    // ========================================================================
    if (warp_active) {
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
}

// Template instantiation
template __global__ void wmma_gemm_kernel_coop<8, 4, 2>(
    const __half*, const __half*, float*, int, int, int);

#endif // WMMA_KERNEL_VARIANTS_HPP


