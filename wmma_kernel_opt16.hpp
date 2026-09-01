// ============================================================================
// 256x128 FP16-input / FP32-accumulate WMMA experiment for gfx1151
//
// This ports the block and wave geometry of adelj_wmma_samples/wmma_opt_5,
// while retaining this project's row-major inputs, gfx1151 FP32 accumulator
// intrinsic, and FP32 output contract.  Keep it separate from matmul_opt until
// it has passed the record benchmark's correctness and repeatability gates.
// ============================================================================

#ifndef WMMA_KERNEL_OPT16_HPP
#define WMMA_KERNEL_OPT16_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

typedef _Float16 half8_opt16 __attribute__((ext_vector_type(8)));

struct Opt16Config {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
    static constexpr int NWARPS = WARPS_M * WARPS_N;
    static constexpr int WARP_TILE_M = 4;
    static constexpr int WARP_TILE_N = 4;
    static constexpr int BLOCK_M = WARPS_M * WARP_TILE_M * WMMA_M;
    static constexpr int BLOCK_N = WARPS_N * WARP_TILE_N * WMMA_N;
    static constexpr int BLOCK_K = WMMA_K;
    static constexpr int LDS_STRIDE_A = BLOCK_K + 8;
    static constexpr int LDS_STRIDE_B = BLOCK_K + 8;
    static constexpr int NUM_THREADS = NWARPS * WARP_SIZE;
};

template<typename cfg>
__device__ __forceinline__ void opt16_load_a(
    __half a_lds[][cfg::LDS_STRIDE_A],
    const __half* __restrict__ a,
    int block_m,
    int k_offset,
    int tid,
    int m,
    int k_size
) {
    const half8_opt16 zero = {0, 0, 0, 0, 0, 0, 0, 0};

    // 256x16 is 512 half8 vectors: every thread loads two vectors.
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int vector_index = tid + pass * cfg::NUM_THREADS;
        const int row = vector_index >> 1;
        const int k_local = (vector_index & 1) << 3;
        half8_opt16 value = zero;
        if (block_m + row < m && k_offset + k_local + 8 <= k_size) {
            value = *reinterpret_cast<const half8_opt16*>(
                a + (block_m + row) * k_size + k_offset + k_local);
        }
        *reinterpret_cast<half8_opt16*>(&a_lds[row][k_local]) = value;
    }
}

template<typename cfg>
__device__ __forceinline__ void opt16_load_b(
    __half b_lds[][cfg::LDS_STRIDE_B],
    const __half* __restrict__ b,
    int block_n,
    int k_offset,
    int tid,
    int n,
    int k_size
) {
    // 16x128 is 256 half8 vectors: every thread owns one contiguous global
    // load, then scatters it into the transposed [N][K] LDS layout expected by
    // the verified gfx1151 fragment loader.
    const int k_local = tid >> 4;
    const int n_local = (tid & 15) << 3;
    const int global_k = k_offset + k_local;
    const int global_n = block_n + n_local;
    half8_opt16 value = {0, 0, 0, 0, 0, 0, 0, 0};
    if (global_k < k_size && global_n + 7 < n) {
        value = *reinterpret_cast<const half8_opt16*>(b + global_k * n + global_n);
    }

    union {
        half8_opt16 vector;
        _Float16 element[8];
    } unpacked;
    unpacked.vector = value;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b_lds[n_local + i][k_local] =
            *reinterpret_cast<__half*>(&unpacked.element[i]);
    }
}

__launch_bounds__(Opt16Config::NUM_THREADS)
__global__ void wmma_gemm_kernel_opt16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    const int m,
    const int n,
    const int k_size
) {
    using cfg = Opt16Config;

    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const int half_lane = lane & 15;
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

    opt16_load_a<cfg>(a_lds[0], a, block_m, 0, tid, m, k_size);
    opt16_load_b<cfg>(b_lds[0], b, block_n, 0, tid, n, k_size);
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        const int next = 1 - current;
        const bool has_next = k + cfg::BLOCK_K < k_size;

        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, row_major> a_frag[cfg::WARP_TILE_M];
        fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, col_major> b_frag[cfg::WARP_TILE_N];

        #pragma unroll
        for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
            load_matrix_sync_lds(
                a_frag[tile_m],
                &a_lds[current][warp_m_base + tile_m * cfg::WMMA_M][0],
                cfg::LDS_STRIDE_A);
        }
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            load_matrix_sync_lds_b_transposed(
                b_frag[tile_n],
                &b_lds[current][warp_n_base + tile_n * cfg::WMMA_N][0],
                cfg::LDS_STRIDE_B);
        }

        if (has_next) {
            opt16_load_a<cfg>(
                a_lds[next], a, block_m, k + cfg::BLOCK_K, tid, m, k_size);
            opt16_load_b<cfg>(
                b_lds[next], b, block_n, k + cfg::BLOCK_K, tid, n, k_size);
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

    const int row_parity = lane >> 4;
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            const int tile_row =
                block_m + warp_m_base + tile_m * cfg::WMMA_M;
            const int column =
                block_n + warp_n_base + tile_n * cfg::WMMA_N + half_lane;
            if (column < n) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = tile_row + (i << 1) + row_parity;
                    if (row < m) {
                        c[row * n + column] = accum[tile_m][tile_n].x[i];
                    }
                }
            }
        }
    }
}

#endif
