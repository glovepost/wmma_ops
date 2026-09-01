// ============================================================================
// Low-live-state K=32 specialization for the 4096-cubed record shape.
//
// A 128x64 workgroup tile keeps four FP32 WMMA accumulators live per wave,
// rather than the eight used by the 128x128 record kernel.  All 256 threads
// issue exactly two A half8 loads and one B half8 load per K=32 panel.  The
// next panel is fetched between independent WMMAs, then published to the
// alternate LDS buffer before the single workgroup barrier for that panel.
//
// The raw gfx1151 builtin avoids wrapper-created fragment copies.  Inputs keep
// the public row-major FP16 contract and accumulation/output remain FP32.
// ============================================================================

#ifndef WMMA_KERNEL_OPT_128X64_K32_HPP
#define WMMA_KERNEL_OPT_128X64_K32_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

using raw_f16_128x64 = _Float16;
using raw_half8_128x64 =
    raw_f16_128x64 __attribute__((ext_vector_type(8)));
using raw_half16_128x64 =
    raw_f16_128x64 __attribute__((ext_vector_type(16)));
using raw_float8_128x64 = float __attribute__((ext_vector_type(8)));

struct Opt128x64K32Config {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
    static constexpr int WARP_TILE_M = 2;
    static constexpr int WARP_TILE_N = 2;
    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 64;
    static constexpr int BLOCK_K = 32;
    static constexpr int LDS_STRIDE = BLOCK_K + 8;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * WARP_SIZE;
};

__device__ __forceinline__ raw_half16_128x64 load_raw_wmma_fragment_128x64(
    const raw_f16_128x64* source
) {
    const raw_half8_128x64 low =
        *reinterpret_cast<const raw_half8_128x64*>(source);
    const raw_half8_128x64 high =
        *reinterpret_cast<const raw_half8_128x64*>(source + 8);
    raw_half16_128x64 result;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        result[i] = low[i];
        result[i + 8] = high[i];
    }
    return result;
}

template<typename cfg>
__device__ __forceinline__ void stage_raw_panel_128x64(
    raw_f16_128x64 a_lds[][cfg::LDS_STRIDE],
    raw_f16_128x64 b_lds[][cfg::LDS_STRIDE],
    int tid,
    raw_half8_128x64 a_low,
    raw_half8_128x64 a_high,
    raw_half8_128x64 b_value
) {
    const int a_row = tid >> 2;
    const int a_k = (tid & 3) << 3;
    *reinterpret_cast<raw_half8_128x64*>(&a_lds[a_row][a_k]) = a_low;
    *reinterpret_cast<raw_half8_128x64*>(&a_lds[a_row + 64][a_k]) =
        a_high;

    const int b_k = tid >> 3;
    const int b_n = (tid & 7) << 3;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        b_lds[b_n + i][b_k] = b_value[i];
    }
}

template<typename cfg>
__device__ __forceinline__ void load_raw_operands_128x64(
    raw_half16_128x64 (&a_frag)[cfg::WARP_TILE_M],
    raw_half16_128x64 (&b_frag)[cfg::WARP_TILE_N],
    const raw_f16_128x64 a_lds[][cfg::LDS_STRIDE],
    const raw_f16_128x64 b_lds[][cfg::LDS_STRIDE],
    int warp_m_base,
    int warp_n_base,
    int k_subtile,
    int lane
) {
    const int lane16 = lane & 15;
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        a_frag[tile_m] = load_raw_wmma_fragment_128x64(
            &a_lds[warp_m_base + tile_m * cfg::WMMA_M + lane16]
                  [k_subtile]);
    }
    #pragma unroll
    for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
        b_frag[tile_n] = load_raw_wmma_fragment_128x64(
            &b_lds[warp_n_base + tile_n * cfg::WMMA_N + lane16]
                  [k_subtile]);
    }
}

__launch_bounds__(Opt128x64K32Config::NUM_THREADS, 2)
__global__ void wmma_gemm_kernel_opt_128x64_k32(
    const raw_f16_128x64* __restrict__ a,
    const raw_f16_128x64* __restrict__ b,
    float* __restrict__ c,
    const int n,
    const int k_size
) {
    using cfg = Opt128x64K32Config;

    __shared__ raw_f16_128x64
        a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE];
    __shared__ raw_f16_128x64
        b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE];

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
    const int b_k = tid >> 3;
    const int b_n = (tid & 7) << 3;

    raw_float8_128x64 accum[2][2] = {
        {raw_float8_128x64{}, raw_float8_128x64{}},
        {raw_float8_128x64{}, raw_float8_128x64{}}
    };

    raw_half8_128x64 next_a_low =
        *reinterpret_cast<const raw_half8_128x64*>(
            a + (block_m + a_row) * k_size + a_k);
    raw_half8_128x64 next_a_high =
        *reinterpret_cast<const raw_half8_128x64*>(
            a + (block_m + a_row + 64) * k_size + a_k);
    raw_half8_128x64 next_b =
        *reinterpret_cast<const raw_half8_128x64*>(
            b + b_k * n + block_n + b_n);
    stage_raw_panel_128x64<cfg>(
        a_lds[0], b_lds[0], tid, next_a_low, next_a_high, next_b);
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        raw_half16_128x64 a_frag[2];
        raw_half16_128x64 b_frag[2];
        load_raw_operands_128x64<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, 0, lane);

        accum[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[0], b_frag[0], accum[0][0]);
        const bool has_next = k + cfg::BLOCK_K < k_size;
        if (has_next) {
            next_a_low = *reinterpret_cast<const raw_half8_128x64*>(
                a + (block_m + a_row) * k_size + k + cfg::BLOCK_K + a_k);
        }
        accum[0][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[0], b_frag[1], accum[0][1]);
        if (has_next) {
            next_a_high = *reinterpret_cast<const raw_half8_128x64*>(
                a + (block_m + a_row + 64) * k_size + k
                    + cfg::BLOCK_K + a_k);
        }
        accum[1][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[1], b_frag[0], accum[1][0]);
        if (has_next) {
            next_b = *reinterpret_cast<const raw_half8_128x64*>(
                b + (k + cfg::BLOCK_K + b_k) * n + block_n + b_n);
        }
        accum[1][1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            a_frag[1], b_frag[1], accum[1][1]);

        load_raw_operands_128x64<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, cfg::WMMA_K, lane);
        #pragma unroll
        for (int tile_m = 0; tile_m < 2; ++tile_m) {
            #pragma unroll
            for (int tile_n = 0; tile_n < 2; ++tile_n) {
                accum[tile_m][tile_n] =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        a_frag[tile_m], b_frag[tile_n],
                        accum[tile_m][tile_n]);
            }
        }

        if (has_next) {
            const int next = 1 - current;
            stage_raw_panel_128x64<cfg>(
                a_lds[next], b_lds[next], tid,
                next_a_low, next_a_high, next_b);
        }
        __syncthreads();
        current = 1 - current;
    }

    const int column_lane = lane & 15;
    const int row_parity = lane >> 4;
    #pragma unroll
    for (int tile_m = 0; tile_m < 2; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < 2; ++tile_n) {
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
