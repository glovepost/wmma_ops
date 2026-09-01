// ============================================================================
// Raw-builtin 128x128x32 record specialization for gfx1151.
//
// This preserves the high-reuse 128x128 output tile while using the same
// low-copy raw vector style as the 128x64 experiment.  Two A and two B half8
// loads are assigned to every thread for each K=32 panel.  Eight independent
// FP32 accumulator chains remain live per wave; operand fragments are reused
// between the two K=16 slices instead of being materialized through the
// rocWMMA compatibility wrappers.
// ============================================================================

#ifndef WMMA_KERNEL_OPT_RAW_K32_HPP
#define WMMA_KERNEL_OPT_RAW_K32_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using raw_f16_k32 = _Float16;
using raw_half8_k32 = raw_f16_k32 __attribute__((ext_vector_type(8)));
using raw_half16_k32 = raw_f16_k32 __attribute__((ext_vector_type(16)));
using raw_float8_k32 = float __attribute__((ext_vector_type(8)));

struct OptRawK32Config {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
    static constexpr int WARP_TILE_M = 2;
    static constexpr int WARP_TILE_N = 4;
    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 128;
    static constexpr int BLOCK_K = 32;
    static constexpr int LDS_STRIDE = BLOCK_K + 8;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * WARP_SIZE;
};

__device__ __forceinline__ raw_half16_k32 load_raw_fragment_k32(
    const raw_f16_k32* source
) {
    const raw_half8_k32 low =
        *reinterpret_cast<const raw_half8_k32*>(source);
    const raw_half8_k32 high =
        *reinterpret_cast<const raw_half8_k32*>(source + 8);
    raw_half16_k32 result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result[i] = low[i];
        result[i + 8] = high[i];
    }
    return result;
}

template<typename cfg>
__device__ __forceinline__ void stage_raw_panel_k32(
    raw_f16_k32 a_lds[][cfg::LDS_STRIDE],
    raw_f16_k32 b_lds[][cfg::LDS_STRIDE],
    int tid,
    raw_half8_k32 a_low,
    raw_half8_k32 a_high,
    raw_half8_k32 b_low,
    raw_half8_k32 b_high
) {
    const int a_row = tid >> 2;
    const int a_k = (tid & 3) << 3;
    *reinterpret_cast<raw_half8_k32*>(&a_lds[a_row][a_k]) = a_low;
    *reinterpret_cast<raw_half8_k32*>(&a_lds[a_row + 64][a_k]) = a_high;

    const int b_k = tid >> 4;
    const int b_n = (tid & 15) << 3;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b_lds[b_n + i][b_k] = b_low[i];
        b_lds[b_n + i][b_k + 16] = b_high[i];
    }
}

template<typename cfg>
__device__ __forceinline__ void load_raw_operands_k32(
    raw_half16_k32 (&a_frag)[cfg::WARP_TILE_M],
    raw_half16_k32 (&b_frag)[cfg::WARP_TILE_N],
    const raw_f16_k32 a_lds[][cfg::LDS_STRIDE],
    const raw_f16_k32 b_lds[][cfg::LDS_STRIDE],
    int warp_m_base,
    int warp_n_base,
    int k_subtile,
    int lane
) {
    const int lane16 = lane & 15;
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        a_frag[tile_m] = load_raw_fragment_k32(
            &a_lds[warp_m_base + tile_m * cfg::WMMA_M + lane16]
                  [k_subtile]);
    }
    #pragma unroll
    for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
        b_frag[tile_n] = load_raw_fragment_k32(
            &b_lds[warp_n_base + tile_n * cfg::WMMA_N + lane16]
                  [k_subtile]);
    }
}

__launch_bounds__(OptRawK32Config::NUM_THREADS)
__global__ void wmma_gemm_kernel_opt_raw_k32(
    const raw_f16_k32* __restrict__ a,
    const raw_f16_k32* __restrict__ b,
    float* __restrict__ c,
    const int n,
    const int k_size
) {
    using cfg = OptRawK32Config;
    __shared__ raw_f16_k32 a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE];
    __shared__ raw_f16_k32 b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp >> 1;
    const int warp_n = warp & 1;
    const int warp_m_base = warp_m * cfg::WARP_TILE_M * cfg::WMMA_M;
    const int warp_n_base = warp_n * cfg::WARP_TILE_N * cfg::WMMA_N;
    const int block_m = blockIdx.y * cfg::BLOCK_M;
    const int block_n = blockIdx.x * cfg::BLOCK_N;
    const int a_row = tid >> 2;
    const int a_k = (tid & 3) << 3;
    const int b_k = tid >> 4;
    const int b_n = (tid & 15) << 3;

    raw_float8_k32 accum[2][4] = {};

    raw_half8_k32 next_a_low =
        *reinterpret_cast<const raw_half8_k32*>(
            a + (block_m + a_row) * k_size + a_k);
    raw_half8_k32 next_a_high =
        *reinterpret_cast<const raw_half8_k32*>(
            a + (block_m + a_row + 64) * k_size + a_k);
    raw_half8_k32 next_b_low =
        *reinterpret_cast<const raw_half8_k32*>(
            b + b_k * n + block_n + b_n);
    raw_half8_k32 next_b_high =
        *reinterpret_cast<const raw_half8_k32*>(
            b + (b_k + 16) * n + block_n + b_n);
    stage_raw_panel_k32<cfg>(a_lds[0], b_lds[0], tid,
                             next_a_low, next_a_high,
                             next_b_low, next_b_high);
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        raw_half16_k32 a_frag[2];
        raw_half16_k32 b_frag[4];
        load_raw_operands_k32<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, 0, lane);

        const bool has_next = k + cfg::BLOCK_K < k_size;
        accum[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[0], b_frag[0], accum[0][0]);
        if (has_next) {
            next_a_low = *reinterpret_cast<const raw_half8_k32*>(
                a + (block_m + a_row) * k_size + k + cfg::BLOCK_K + a_k);
        }
        accum[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[0], b_frag[1], accum[0][1]);
        if (has_next) {
            next_a_high = *reinterpret_cast<const raw_half8_k32*>(
                a + (block_m + a_row + 64) * k_size + k
                    + cfg::BLOCK_K + a_k);
        }
        accum[0][2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[0], b_frag[2], accum[0][2]);
        if (has_next) {
            next_b_low = *reinterpret_cast<const raw_half8_k32*>(
                b + (k + cfg::BLOCK_K + b_k) * n + block_n + b_n);
        }
        accum[0][3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[0], b_frag[3], accum[0][3]);
        if (has_next) {
            next_b_high = *reinterpret_cast<const raw_half8_k32*>(
                b + (k + cfg::BLOCK_K + b_k + 16) * n + block_n + b_n);
        }
        #pragma unroll
        for (int tile_n = 0; tile_n < 4; ++tile_n) {
            accum[1][tile_n] =
                __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                    a_frag[1], b_frag[tile_n], accum[1][tile_n]);
        }

        load_raw_operands_k32<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, cfg::WMMA_K, lane);
        #pragma unroll
        for (int tile_m = 0; tile_m < 2; ++tile_m) {
            #pragma unroll
            for (int tile_n = 0; tile_n < 4; ++tile_n) {
                accum[tile_m][tile_n] =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        a_frag[tile_m], b_frag[tile_n],
                        accum[tile_m][tile_n]);
            }
        }

        if (has_next) {
            const int next = 1 - current;
            stage_raw_panel_k32<cfg>(a_lds[next], b_lds[next], tid,
                                     next_a_low, next_a_high,
                                     next_b_low, next_b_high);
        }
        __syncthreads();
        current = 1 - current;
    }

    const int column_lane = lane & 15;
    const int row_parity = lane >> 4;
    #pragma unroll
    for (int tile_m = 0; tile_m < 2; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < 4; ++tile_n) {
            const int tile_row =
                block_m + warp_m_base + tile_m * cfg::WMMA_M;
            const int column = block_n + warp_n_base
                + tile_n * cfg::WMMA_N + column_lane;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = tile_row + (i << 1) + row_parity;
                c[row * n + column] = accum[tile_m][tile_n][i];
            }
        }
    }
}

#endif
