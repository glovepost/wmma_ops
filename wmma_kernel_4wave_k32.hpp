// ============================================================================
// Four-wave 128x128x32 specialization for gfx1151.
//
// rocBLAS 7.14 selects this macro-tile and four-wave geometry for the tuned
// FP16-output path on gfx1151.  This clean-room variant preserves this
// project's stricter FP32 output contract and verified rocWMMA fragment
// layout.  Each wave owns a 4x4 WMMA accumulator tile.  B fragments are
// streamed one at a time to keep the 16 FP32 accumulator fragments resident
// without spilling.
// ============================================================================

#ifndef WMMA_KERNEL_4WAVE_K32_HPP
#define WMMA_KERNEL_4WAVE_K32_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

using half8_4wave = _Float16 __attribute__((ext_vector_type(8)));

struct FourWaveK32Config {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_M = 2;
    static constexpr int WARPS_N = 2;
    static constexpr int NWARPS = WARPS_M * WARPS_N;
    static constexpr int WARP_TILE_M = 4;
    static constexpr int WARP_TILE_N = 4;
    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 128;
    static constexpr int BLOCK_K = 32;
    // The rocBLAS-selected solution reports one LDS buffer and 17,408 bytes:
    // an unpadded 128x32 A panel plus a 128x32 B panel with four-half padding.
    static constexpr int LDS_STRIDE_A = BLOCK_K;
    static constexpr int LDS_STRIDE_B = BLOCK_K + 4;
    static constexpr int NUM_THREADS = NWARPS * WARP_SIZE;
};

template<typename cfg>
__device__ __forceinline__ void fourwave_stage_a(
    __half a_lds[][cfg::LDS_STRIDE_A],
    int row,
    const half8_4wave (&values)[4]
) {
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        *reinterpret_cast<half8_4wave*>(&a_lds[row][vector * 8]) =
            values[vector];
    }
}

template<typename cfg>
__device__ __forceinline__ void fourwave_stage_b(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int k_lane,
    int n_base,
    const half8_4wave (&values)[4]
) {
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        union {
            half8_4wave packed;
            _Float16 element[8];
        } unpacked;
        unpacked.packed = values[vector];
        #pragma unroll
        for (int n_local = 0; n_local < 8; ++n_local) {
            b_lds[n_base + n_local][vector * 8 + k_lane] =
                *reinterpret_cast<__half*>(&unpacked.element[n_local]);
        }
    }
}

__launch_bounds__(FourWaveK32Config::NUM_THREADS)
__global__ void wmma_gemm_kernel_4wave_k32(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int n,
    int k_size
) {
    using cfg = FourWaveK32Config;
    __shared__ __half a_lds[cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const int k_lane = tid >> 4;
    const int n_base = (tid & 15) << 3;
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

    half8_4wave initial_a[4];
    half8_4wave initial_b[4];
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        initial_a[vector] = *reinterpret_cast<const half8_4wave*>(
            a + (block_m + tid) * k_size + vector * 8);
        initial_b[vector] = *reinterpret_cast<const half8_4wave*>(
            b + (vector * 8 + k_lane) * n + block_n + n_base);
    }
    fourwave_stage_a<cfg>(a_lds, tid, initial_a);
    fourwave_stage_b<cfg>(b_lds, k_lane, n_base, initial_b);
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        const bool has_next = k + cfg::BLOCK_K < k_size;
        half8_4wave prefetched_a[4] = {};
        half8_4wave prefetched_b[4] = {};
        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, row_major> a_frag[cfg::WARP_TILE_M];

        #pragma unroll
        for (int subtile = 0; subtile < 2; ++subtile) {
            #pragma unroll
            for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
                load_matrix_sync_lds(
                    a_frag[tile_m],
                    &a_lds[warp_m_base + tile_m * cfg::WMMA_M]
                          [subtile * cfg::WMMA_K],
                    cfg::LDS_STRIDE_A);
            }

            #pragma unroll
            for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
                fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                         __half, col_major> b_frag;
                load_matrix_sync_lds_b_transposed(
                    b_frag,
                    &b_lds[warp_n_base + tile_n * cfg::WMMA_N]
                          [subtile * cfg::WMMA_K],
                    cfg::LDS_STRIDE_B);

                #pragma unroll
                for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
                    mma_sync(accum[tile_m][tile_n], a_frag[tile_m], b_frag,
                             accum[tile_m][tile_n]);
                    if (has_next && subtile == 0) {
                        const int load_index = tile_n * cfg::WARP_TILE_M
                            + tile_m;
                        if (load_index < 4) {
                            prefetched_a[load_index] =
                                *reinterpret_cast<const half8_4wave*>(
                                    a + (block_m + tid) * k_size + k
                                        + cfg::BLOCK_K + load_index * 8);
                        } else if (load_index < 8) {
                            const int vector = load_index - 4;
                            prefetched_b[vector] =
                                *reinterpret_cast<const half8_4wave*>(
                                    b + (k + cfg::BLOCK_K + vector * 8
                                         + k_lane) * n
                                        + block_n + n_base);
                        }
                    }
                }
            }
        }

        if (has_next) {
            // The single-buffer schedule requires every wave to finish its
            // local reads before any wave overwrites the panels, then a
            // second rendezvous before the next iteration consumes them.
            __syncthreads();
            fourwave_stage_a<cfg>(a_lds, tid, prefetched_a);
            fourwave_stage_b<cfg>(b_lds, k_lane, n_base, prefetched_b);
            __syncthreads();
        }
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
