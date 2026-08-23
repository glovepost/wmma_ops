/*
 * Wave-private A / double-buffered B experiment for gfx1151.
 *
 * The 256x128 block-packed leader stages one 12 KiB K16 tile and needs two
 * workgroup barriers per step: one before overwriting it and one after.  This
 * kernel removes the overwrite dependency without allocating two complete
 * tiles.  Each of the eight waves owns a private 64x16 A tile (p4), while the
 * workgroup ping-pongs two shared 128x16 B tiles (p8):
 *
 *   8 * 64 * (16 + 4) * 2 bytes = 20 KiB A
 *   2 * 128 * (16 + 8) * 2 bytes = 12 KiB B
 *
 * The exact 32 KiB total retains two workgroups per CU.  The two N waves for
 * each output-row band intentionally fetch duplicate A data, trading extra
 * cache traffic for one barrier per K16.  A refills are written after that
 * wave's final use; B refills target the inactive buffer and are published by
 * the sole barrier.
 */
#ifndef ROCM_WMMA_GEMM_WAVEPRIVATE_A_HPP
#define ROCM_WMMA_GEMM_WAVEPRIVATE_A_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_WPA_PADDING_A
#define WMMA_WPA_PADDING_A 4
#endif
#ifndef WMMA_WPA_PADDING_B
#define WMMA_WPA_PADDING_B 8
#endif
#ifndef WMMA_WPA_SWIZZLE
#define WMMA_WPA_SWIZZLE 16
#endif

struct waveprivate_a_gemm
{
    __global__ __launch_bounds__(256)
        __attribute__((amdgpu_waves_per_eu(2, 8))) static void run(
            half* __restrict__ C,
            const half* __restrict__ A,
            const half* __restrict__ B,
            int M,
            int N,
            int K)
    {
        constexpr int block_m = 256;
        constexpr int block_n = 128;
        constexpr int block_k = wmma_tile;
        constexpr int wave_rows = 64;
        constexpr int stride_a = block_k + WMMA_WPA_PADDING_A;
        constexpr int stride_b = block_k + WMMA_WPA_PADDING_B;
        constexpr int a_wave_elements = wave_rows * stride_a;
        constexpr int b_buffer_elements = block_n * stride_b;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        static_assert(8 * a_wave_elements + 2 * b_buffer_elements == 16384,
                      "wave-private architecture must remain exactly 32 KiB");

        __shared__ half a_lds[8 * a_wave_elements];
        __shared__ half b_lds[2 * b_buffer_elements];

        const int tid = static_cast<int>(threadIdx.x);
        const int lane = tid & (warp_size - 1);
        const int half_lane = lane & 15;
        const int half_wave = lane >> 4;
        const int wave = tid / warp_size;
        const int warp_row = wave >> 1;
        const int warp_col = wave & 1;

        const int grid_m = M / block_m;
        const int grid_n = N / block_n;
        int block_row = 0;
        int block_col = 0;
        tile_mapper<block_m,
                    block_n,
                    m_layout::col_major,
                    m_layout::row_major,
                    WMMA_WPA_SWIZZLE>()
            .map_tile(static_cast<int>(blockIdx.x),
                      grid_m,
                      grid_n,
                      &block_row,
                      &block_col);

        const int k_tiles = K / block_k;
        const int block_m_index = block_row / block_m;
        const int block_n_index = block_col / block_n;
        const half* a_tile = A
            + static_cast<size_t>(block_m_index) * k_tiles * a_tile_elements;
        const half* b_tile = B
            + static_cast<size_t>(block_n_index) * k_tiles * b_tile_elements;

        const int a_wave_base = wave * a_wave_elements;
        const int b_row = tid >> 1;
        const int b_half = (tid & 1) * 8;

        // Seed private A in the same per-WMMA chunks used by the hot loop.
        const u16x8* a_vectors = reinterpret_cast<const u16x8*>(a_tile);
        #pragma unroll
        for(int wm = 0; wm < 4; ++wm)
        {
            const int vector_index
                = (warp_row * wave_rows + wm * wmma_tile) * 2 + lane;
            const u16x8 staged = a_vectors[vector_index];
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            const int row = wm * wmma_tile + (lane >> 1);
            const int half_offset = (lane & 1) * 8;
            *reinterpret_cast<u16x8*>(
                a_lds + a_wave_base + row * stride_a + half_offset) = staged;
        }

        const u16x8* b_vectors = reinterpret_cast<const u16x8*>(b_tile);
        const u16x8 seed_b = b_vectors[tid];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(
            b_lds + b_row * stride_b + b_half) = seed_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];
        const int warp_n_base = warp_col * 4 * wmma_tile;

        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int current_buffer = k_tile & 1;
            const int next_buffer = 1 - current_buffer;
            const bool has_next = k_tile + 1 < k_tiles;
            const half* next_a = has_next
                ? a_tile + (k_tile + 1) * a_tile_elements
                : nullptr;
            const half* next_b = has_next
                ? b_tile + (k_tile + 1) * b_tile_elements
                : nullptr;

            fragment<half, wmma_tile> b_frag[4];
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                const half* source = b_lds
                    + current_buffer * b_buffer_elements
                    + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag[wn], source, block_k, stride_b);
            }

            u16x8 staged_b;
            if(has_next)
            {
                const u16x8* next_b_vectors
                    = reinterpret_cast<const u16x8*>(next_b);
                staged_b = next_b_vectors[tid];
            }

            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                u16x8 staged_a;
                if(has_next)
                {
                    const u16x8* next_a_vectors
                        = reinterpret_cast<const u16x8*>(next_a);
                    const int vector_index
                        = (warp_row * wave_rows + wm * wmma_tile) * 2 + lane;
                    staged_a = next_a_vectors[vector_index];
                }

                fragment<half, wmma_tile> a_frag;
                const half* source = a_lds
                    + a_wave_base
                    + (wm * wmma_tile + half_lane) * stride_a;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag, source, stride_a, block_k);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    if(wn < 2)
                        wmma<false>(a_frag, b_frag[wn], accum[wm][wn]);
                    else
                        wmma<true>(a_frag, b_frag[wn], accum[wm][wn - 2]);
                }

                if(has_next)
                {
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    const int row = wm * wmma_tile + (lane >> 1);
                    const int half_offset = (lane & 1) * 8;
                    *reinterpret_cast<u16x8*>(
                        a_lds + a_wave_base + row * stride_a + half_offset)
                        = staged_a;

                    if(wm == 0)
                    {
                        *reinterpret_cast<u16x8*>(
                            b_lds + next_buffer * b_buffer_elements
                                + b_row * stride_b + b_half) = staged_b;
                    }
                }
            }

            if(has_next)
            {
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
                __syncthreads();
            }
        }

        const int warp_m_base = warp_row * 4 * wmma_tile;
        #pragma unroll
        for(int wm = 0; wm < 4; ++wm)
        {
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                if(wn < 2)
                    store_matrix<m_layout::row_major, false, false>(
                        C,
                        accum[wm][wn],
                        block_row + warp_m_base + wm * wmma_tile + half_wave,
                        block_col + warp_n_base + wn * wmma_tile + half_lane,
                        M,
                        N);
                else
                    store_matrix<m_layout::row_major, false, true>(
                        C,
                        accum[wm][wn - 2],
                        block_row + warp_m_base + wm * wmma_tile + half_wave,
                        block_col + warp_n_base + wn * wmma_tile + half_lane,
                        M,
                        N);
            }
        }
    }
};

} // namespace rocm_wmma_gemm

#endif
