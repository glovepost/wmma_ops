// ============================================================================
// Direct row-major LDS experiment for the 4096-cubed record shape.
//
// Unlike matmul_opt, B remains row-major in LDS.  This removes the scalar
// global-to-LDS transpose and its address/control overhead; each wave gathers
// its B fragment down the K dimension.  The binding deliberately requires
// complete 128x128x16 tiles so the hot loop contains no edge predicates.
// ============================================================================

#ifndef WMMA_KERNEL_DIRECT_LDS_HPP
#define WMMA_KERNEL_DIRECT_LDS_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

typedef _Float16 half8_direct __attribute__((ext_vector_type(8)));

struct DirectLdsConfig {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
    static constexpr int NWARPS = WARPS_M * WARPS_N;
    static constexpr int WARP_TILE_M = 2;
    static constexpr int WARP_TILE_N = 4;
    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 128;
    static constexpr int BLOCK_K = 16;
    static constexpr int NUM_THREADS = NWARPS * WARP_SIZE;
};

template<typename cfg>
__device__ __forceinline__ void direct_stage_tile(
    __half a_lds[][cfg::BLOCK_K],
    __half b_lds[][cfg::BLOCK_N],
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int block_m,
    int block_n,
    int k_offset,
    int tid,
    int n,
    int k_size
) {
    // A and B tiles both contain exactly 256 aligned half8 vectors.
    const int a_row = tid >> 1;
    const int a_k = (tid & 1) << 3;
    const int b_k = tid >> 4;
    const int b_n = (tid & 15) << 3;

    *reinterpret_cast<half8_direct*>(&a_lds[a_row][a_k]) =
        *reinterpret_cast<const half8_direct*>(
            a + (block_m + a_row) * k_size + k_offset + a_k);
    *reinterpret_cast<half8_direct*>(&b_lds[b_k][b_n]) =
        *reinterpret_cast<const half8_direct*>(
            b + (k_offset + b_k) * n + block_n + b_n);
}

template<typename cfg>
__device__ __forceinline__ void direct_load_b_fragment(
    fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
             __half, col_major>& frag,
    const __half b_lds[][cfg::BLOCK_N],
    int column_base
) {
    const int column = column_base + (threadIdx.x & 15);
    #pragma unroll
    for (int k = 0; k < cfg::BLOCK_K; ++k) {
        frag.x[k] = b_lds[k][column];
    }
}

__launch_bounds__(DirectLdsConfig::NUM_THREADS)
__global__ void wmma_gemm_kernel_direct_lds(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    const int n,
    const int k_size
) {
    using cfg = DirectLdsConfig;

    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::BLOCK_K];
    __shared__ __half b_lds[2][cfg::BLOCK_K][cfg::BLOCK_N];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const int warp_m = warp / cfg::WARPS_N;
    const int warp_n = warp % cfg::WARPS_N;
    const int warp_m_base = warp_m * cfg::WARP_TILE_M * cfg::WMMA_M;
    const int warp_n_base = warp_n * cfg::WARP_TILE_N * cfg::WMMA_N;
    const int block_m = blockIdx.y * cfg::BLOCK_M;
    const int block_n = blockIdx.x * cfg::BLOCK_N;

    fragment<accumulator, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, float>
        accum[cfg::WARP_TILE_M][cfg::WARP_TILE_N];
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            fill_fragment(accum[tile_m][tile_n], 0.0f);
        }
    }

    direct_stage_tile<cfg>(
        a_lds[0], b_lds[0], a, b, block_m, block_n, 0, tid, n, k_size);
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        const int next = 1 - current;
        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, row_major> a_frag[cfg::WARP_TILE_M];
        fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, col_major> b_frag[cfg::WARP_TILE_N];

        #pragma unroll
        for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
            load_matrix_sync_lds(
                a_frag[tile_m],
                &a_lds[current][warp_m_base + tile_m * cfg::WMMA_M][0],
                cfg::BLOCK_K);
        }
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            direct_load_b_fragment<cfg>(
                b_frag[tile_n], b_lds[current],
                warp_n_base + tile_n * cfg::WMMA_N);
        }

        if (k + cfg::BLOCK_K < k_size) {
            direct_stage_tile<cfg>(
                a_lds[next], b_lds[next], a, b, block_m, block_n,
                k + cfg::BLOCK_K, tid, n, k_size);
        }

        #pragma unroll
        for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
            #pragma unroll
            for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
                mma_sync(accum[tile_m][tile_n], a_frag[tile_m],
                         b_frag[tile_n], accum[tile_m][tile_n]);
            }
        }

        __syncthreads();
        current = next;
    }

    const int column_lane = lane & 15;
    const int row_parity = lane >> 4;
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            const int tile_row =
                block_m + warp_m_base + tile_m * cfg::WMMA_M;
            const int column = block_n + warp_n_base
                + tile_n * cfg::WMMA_N + column_lane;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = tile_row + (i << 1) + row_parity;
                c[row * n + column] = accum[tile_m][tile_n].x[i];
            }
        }
    }
}

#endif
