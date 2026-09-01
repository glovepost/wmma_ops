// ============================================================================
// Single-wave 32x64 record-shape experiment for gfx1151.
//
// Each workgroup is exactly one wave and owns its LDS buffers.  This removes
// inter-wave barrier waiting while retaining a 2x4 WMMA accumulator tile.  It
// deliberately trades the 128x128 kernel's cross-wave operand reuse for many
// independent wave tiles; the 4096-cubed benchmark decides whether cache reuse
// is sufficient to make that trade profitable.
// ============================================================================

#ifndef WMMA_KERNEL_SINGLE_WAVE_HPP
#define WMMA_KERNEL_SINGLE_WAVE_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

using half8_single_wave = _Float16 __attribute__((ext_vector_type(8)));

struct SingleWaveConfig {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_TILE_M = 2;
    static constexpr int WARP_TILE_N = 4;
    static constexpr int BLOCK_M = 32;
    static constexpr int BLOCK_N = 64;
    static constexpr int BLOCK_K = 16;
    static constexpr int LDS_STRIDE_A = BLOCK_K + 8;
    static constexpr int LDS_STRIDE_B = BLOCK_K + 8;
    static constexpr int NUM_THREADS = 32;
};

template<typename cfg>
__device__ __forceinline__ void single_wave_stage(
    __half a_lds[][cfg::LDS_STRIDE_A],
    __half b_lds[][cfg::LDS_STRIDE_B],
    int lane,
    const half8_single_wave (&a_value)[2],
    const half8_single_wave (&b_value)[4]
) {
    *reinterpret_cast<half8_single_wave*>(&a_lds[lane][0]) = a_value[0];
    *reinterpret_cast<half8_single_wave*>(&a_lds[lane][8]) = a_value[1];

    const int b_k = lane & 15;
    const int b_n_base = (lane >> 4) * 32;
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        union {
            half8_single_wave packed;
            _Float16 element[8];
        } unpacked;
        unpacked.packed = b_value[vector];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            b_lds[b_n_base + vector * 8 + i][b_k] =
                *reinterpret_cast<__half*>(&unpacked.element[i]);
        }
    }
}

__launch_bounds__(SingleWaveConfig::NUM_THREADS)
__global__ void wmma_gemm_kernel_single_wave(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int n,
    int k_size
) {
    using cfg = SingleWaveConfig;
    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int lane = threadIdx.x;
    const int lane16 = lane & 15;
    const int block_m = blockIdx.y * cfg::BLOCK_M;
    const int block_n = blockIdx.x * cfg::BLOCK_N;
    const int b_k = lane16;
    const int b_n_base = (lane >> 4) * 32;

    fragment<accumulator, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, float>
        accum[cfg::WARP_TILE_M][cfg::WARP_TILE_N];
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            fill_fragment(accum[tile_m][tile_n], 0.0f);
        }
    }

    half8_single_wave next_a[2];
    half8_single_wave next_b[4];
    const __half* a_source = a + (block_m + lane) * k_size;
    next_a[0] = *reinterpret_cast<const half8_single_wave*>(a_source);
    next_a[1] = *reinterpret_cast<const half8_single_wave*>(a_source + 8);
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        next_b[vector] = *reinterpret_cast<const half8_single_wave*>(
            b + b_k * n + block_n + b_n_base + vector * 8);
    }
    single_wave_stage<cfg>(a_lds[0], b_lds[0], lane, next_a, next_b);
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, row_major> a_frag[2];
        #pragma unroll
        for (int tile_m = 0; tile_m < 2; ++tile_m) {
            load_matrix_sync_lds(
                a_frag[tile_m],
                &a_lds[current][tile_m * cfg::WMMA_M][0],
                cfg::LDS_STRIDE_A);
        }
        const bool has_next = k + cfg::BLOCK_K < k_size;
        #pragma unroll
        for (int tile_n = 0; tile_n < 4; ++tile_n) {
            fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                     __half, col_major> b_frag;
            load_matrix_sync_lds_b_transposed(
                b_frag,
                &b_lds[current][tile_n * cfg::WMMA_N][0],
                cfg::LDS_STRIDE_B);

            mma_sync(accum[0][tile_n], a_frag[0], b_frag,
                     accum[0][tile_n]);
            if (has_next) {
                const int load_index = tile_n * 2;
                if (load_index == 0) {
                    next_a[0] =
                        *reinterpret_cast<const half8_single_wave*>(
                            a + (block_m + lane) * k_size + k
                                + cfg::BLOCK_K);
                } else if (load_index == 2) {
                    next_b[0] =
                        *reinterpret_cast<const half8_single_wave*>(
                            b + (k + cfg::BLOCK_K + b_k) * n
                                + block_n + b_n_base);
                } else if (load_index == 4) {
                    next_b[2] =
                        *reinterpret_cast<const half8_single_wave*>(
                            b + (k + cfg::BLOCK_K + b_k) * n
                                + block_n + b_n_base + 16);
                }
            }

            mma_sync(accum[1][tile_n], a_frag[1], b_frag,
                     accum[1][tile_n]);
            if (has_next) {
                const int load_index = tile_n * 2 + 1;
                if (load_index == 1) {
                    next_a[1] =
                        *reinterpret_cast<const half8_single_wave*>(
                            a + (block_m + lane) * k_size + k
                                + cfg::BLOCK_K + 8);
                } else if (load_index == 3) {
                    next_b[1] =
                        *reinterpret_cast<const half8_single_wave*>(
                            b + (k + cfg::BLOCK_K + b_k) * n
                                + block_n + b_n_base + 8);
                } else if (load_index == 5) {
                    next_b[3] =
                        *reinterpret_cast<const half8_single_wave*>(
                            b + (k + cfg::BLOCK_K + b_k) * n
                                + block_n + b_n_base + 24);
                }
            }
        }

        if (has_next) {
            const int next = 1 - current;
            single_wave_stage<cfg>(
                a_lds[next], b_lds[next], lane, next_a, next_b);
        }
        __syncthreads();
        current = 1 - current;
    }

    const int row_parity = lane >> 4;
    #pragma unroll
    for (int tile_m = 0; tile_m < 2; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < 4; ++tile_n) {
            const int tile_row = block_m + tile_m * cfg::WMMA_M;
            const int column =
                block_n + tile_n * cfg::WMMA_N + lane16;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = tile_row + (i << 1) + row_parity;
                c[row * n + column] = accum[tile_m][tile_n].x[i];
            }
        }
    }
}

#endif
