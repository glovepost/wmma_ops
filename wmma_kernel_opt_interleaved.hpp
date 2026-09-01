// ============================================================================
// Interleaved 128x128 record-shape specialization for gfx1151.
//
// This keeps matmul_opt's validated 2x4 FP32 accumulator tile, but issues the
// next A/B global loads between independent WMMA chains and delays LDS stores
// until those loads can complete.  Complete 128x128x16 tiles are required so
// no edge predicates enter the hot loop.
// ============================================================================

#ifndef WMMA_KERNEL_OPT_INTERLEAVED_HPP
#define WMMA_KERNEL_OPT_INTERLEAVED_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocwmma_patch/rocwmma_gfx1151.hpp"
#include "wmma_kernel_largetile.hpp"

using namespace rocwmma;

typedef _Float16 half8_interleaved __attribute__((ext_vector_type(8)));

template<typename cfg>
__device__ __forceinline__ void interleaved_stage_a(
    __half a_lds[][cfg::LDS_STRIDE_A],
    int row,
    half8_interleaved first,
    half8_interleaved second
) {
    *reinterpret_cast<half8_interleaved*>(&a_lds[row][0]) = first;
    *reinterpret_cast<half8_interleaved*>(&a_lds[row][8]) = second;
}

template<typename cfg>
__device__ __forceinline__ void interleaved_stage_b(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int lane8,
    int n_base,
    half8_interleaved first,
    half8_interleaved second
) {
    union {
        half8_interleaved vector;
        _Float16 element[8];
    } unpacked_first, unpacked_second;
    unpacked_first.vector = first;
    unpacked_second.vector = second;

    #pragma unroll
    for (int n_local = 0; n_local < 8; ++n_local) {
        b_lds[n_base + n_local][lane8] =
            *reinterpret_cast<__half*>(&unpacked_first.element[n_local]);
        b_lds[n_base + n_local][8 + lane8] =
            *reinterpret_cast<__half*>(&unpacked_second.element[n_local]);
    }
}

// B has already been transposed to a row-major [N,K] tensor.  Each loader
// owns one output column, so the two global half8 reads can be published with
// two vector LDS stores instead of a sixteen-element scalar scatter.
template<typename cfg>
__device__ __forceinline__ void interleaved_stage_b_transposed_input(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int column,
    half8_interleaved first,
    half8_interleaved second
) {
    *reinterpret_cast<half8_interleaved*>(&b_lds[column][0]) = first;
    *reinterpret_cast<half8_interleaved*>(&b_lds[column][8]) = second;
}

// Pair adjacent K lanes with a DPP row_xmask operation, then write two halves
// per LDS instruction.  The even lane contributes the low half and its odd
// neighbor contributes the high half.  This preserves B_lds[N][K] while
// replacing sixteen 16-bit stores with eight aligned 32-bit stores per vector
// pair, without consuming the LDS crossbar for the lane exchange.
template<typename cfg>
__device__ __forceinline__ void interleaved_stage_b_packed(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int lane8,
    int n_base,
    half8_interleaved first,
    half8_interleaved second
) {
    union {
        half8_interleaved vector;
        _Float16 element[8];
    } unpacked_first, unpacked_second;
    unpacked_first.vector = first;
    unpacked_second.vector = second;

    #pragma unroll
    for (int n_local = 0; n_local < 8; ++n_local) {
        const uint32_t first_bits = __builtin_bit_cast(
            uint16_t, unpacked_first.element[n_local]);
        const uint32_t second_bits = __builtin_bit_cast(
            uint16_t, unpacked_second.element[n_local]);
        const uint32_t first_adjacent = __builtin_amdgcn_mov_dpp(
            first_bits, 0x161, 0xf, 0xf, false);
        const uint32_t second_adjacent = __builtin_amdgcn_mov_dpp(
            second_bits, 0x161, 0xf, 0xf, false);

        if ((lane8 & 1) == 0) {
            *reinterpret_cast<uint32_t*>(&b_lds[n_base + n_local][lane8]) =
                first_bits | (first_adjacent << 16);
            *reinterpret_cast<uint32_t*>(&b_lds[n_base + n_local][8 + lane8]) =
                second_bits | (second_adjacent << 16);
        }
    }
}

template<bool PACKED_B = false, bool TRANSPOSED_INPUT = false>
__launch_bounds__(OptConfig::NUM_THREADS)
__global__ void wmma_gemm_kernel_opt_interleaved(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    const int n,
    const int k_size
) {
    using cfg = OptConfig;

    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const int cid = tid & (cfg::HALF_BLOCK - 1);
    const bool loads_a = tid < cfg::HALF_BLOCK;
    const int lane8 = cid & 7;
    const int n_base = (cid >> 3) << 3;
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

    if (loads_a) {
        const __half* source = a + (block_m + cid) * k_size;
        interleaved_stage_a<cfg>(
            a_lds[0], cid,
            *reinterpret_cast<const half8_interleaved*>(source),
            *reinterpret_cast<const half8_interleaved*>(source + 8));
    } else {
        half8_interleaved first;
        half8_interleaved second;
        if constexpr (TRANSPOSED_INPUT) {
            const __half* source = b + (block_n + cid) * k_size;
            first = *reinterpret_cast<const half8_interleaved*>(source);
            second = *reinterpret_cast<const half8_interleaved*>(source + 8);
        } else {
            const __half* source = b + lane8 * n + block_n + n_base;
            first = *reinterpret_cast<const half8_interleaved*>(source);
            second = *reinterpret_cast<const half8_interleaved*>(source + 8 * n);
        }
        if constexpr (TRANSPOSED_INPUT) {
            interleaved_stage_b_transposed_input<cfg>(
                b_lds[0], cid, first, second);
        } else if constexpr (PACKED_B) {
            interleaved_stage_b_packed<cfg>(
                b_lds[0], lane8, n_base, first, second);
        } else {
            interleaved_stage_b<cfg>(
                b_lds[0], lane8, n_base, first, second);
        }
    }
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

        half8_interleaved first = {};
        half8_interleaved second = {};

        mma_sync(accum[0][0], a_frag[0], b_frag[0], accum[0][0]);
        if (has_next) {
            if (loads_a) {
                const __half* source =
                    a + (block_m + cid) * k_size + k + cfg::BLOCK_K;
                first = *reinterpret_cast<const half8_interleaved*>(source);
            } else {
                const __half* source;
                if constexpr (TRANSPOSED_INPUT) {
                    source = b + (block_n + cid) * k_size + k
                        + cfg::BLOCK_K;
                } else {
                    source = b + (k + cfg::BLOCK_K + lane8) * n
                        + block_n + n_base;
                }
                first = *reinterpret_cast<const half8_interleaved*>(source);
            }
        }

        mma_sync(accum[0][1], a_frag[0], b_frag[1], accum[0][1]);
        if (has_next) {
            if (loads_a) {
                const __half* source =
                    a + (block_m + cid) * k_size + k + cfg::BLOCK_K + 8;
                second = *reinterpret_cast<const half8_interleaved*>(source);
            } else {
                const __half* source;
                if constexpr (TRANSPOSED_INPUT) {
                    source = b + (block_n + cid) * k_size + k
                        + cfg::BLOCK_K + 8;
                } else {
                    source = b + (k + cfg::BLOCK_K + 8 + lane8) * n
                        + block_n + n_base;
                }
                second = *reinterpret_cast<const half8_interleaved*>(source);
            }
        }

        mma_sync(accum[0][2], a_frag[0], b_frag[2], accum[0][2]);
        mma_sync(accum[0][3], a_frag[0], b_frag[3], accum[0][3]);
        mma_sync(accum[1][0], a_frag[1], b_frag[0], accum[1][0]);
        mma_sync(accum[1][1], a_frag[1], b_frag[1], accum[1][1]);
        mma_sync(accum[1][2], a_frag[1], b_frag[2], accum[1][2]);
        mma_sync(accum[1][3], a_frag[1], b_frag[3], accum[1][3]);

        if (has_next) {
            if (loads_a) {
                interleaved_stage_a<cfg>(a_lds[next], cid, first, second);
            } else {
                if constexpr (TRANSPOSED_INPUT) {
                    interleaved_stage_b_transposed_input<cfg>(
                        b_lds[next], cid, first, second);
                } else if constexpr (PACKED_B) {
                    interleaved_stage_b_packed<cfg>(
                        b_lds[next], lane8, n_base, first, second);
                } else {
                    interleaved_stage_b<cfg>(
                        b_lds[next], lane8, n_base, first, second);
                }
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
