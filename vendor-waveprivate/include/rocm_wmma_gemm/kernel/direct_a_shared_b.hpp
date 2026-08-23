/*
 * Direct-A / shared-B block-prepacked experiment for gfx1151.
 *
 * A fragments are loaded directly from the persistent block/K-major input.
 * The two N waves in each output-row band therefore request the same A rows,
 * relying on vector-cache reuse instead of an LDS round trip.  B remains
 * cooperative and is double-buffered in 12 KiB LDS.  The inactive B buffer is
 * refilled while the current K16 tile computes, leaving one barrier per step.
 */
#ifndef ROCM_WMMA_GEMM_DIRECT_A_SHARED_B_HPP
#define ROCM_WMMA_GEMM_DIRECT_A_SHARED_B_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_DASB_PADDING_B
#define WMMA_DASB_PADDING_B 8
#endif
#ifndef WMMA_DASB_SWIZZLE
#define WMMA_DASB_SWIZZLE 16
#endif

struct direct_a_shared_b_gemm
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
        constexpr int stride_b = block_k + WMMA_DASB_PADDING_B;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        constexpr int b_buffer_elements = block_n * stride_b;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

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
                    WMMA_DASB_SWIZZLE>()
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

        const int b_row = tid >> 1;
        const int b_half = (tid & 1) * 8;
        const u16x8 seed_b
            = reinterpret_cast<const u16x8*>(b_tile)[tid];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(
            b_lds + b_row * stride_b + b_half) = seed_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];
        const int warp_m_base = warp_row * 4 * wmma_tile;
        const int warp_n_base = warp_col * 4 * wmma_tile;

        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int current_buffer = k_tile & 1;
            const int next_buffer = current_buffer ^ 1;
            const bool has_next = k_tile + 1 < k_tiles;

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
                const u16x8* next_b = reinterpret_cast<const u16x8*>(
                    b_tile + (k_tile + 1) * b_tile_elements);
                staged_b = next_b[tid];
            }

            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                fragment<half, wmma_tile> a_frag;
                const half* source = a_tile
                    + k_tile * a_tile_elements
                    + (warp_m_base + wm * wmma_tile + half_lane) * block_k;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag, source, block_k, block_k);
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    if(wn < 2)
                        wmma<false>(a_frag, b_frag[wn], accum[wm][wn]);
                    else
                        wmma<true>(a_frag, b_frag[wn], accum[wm][wn - 2]);
                }
            }

            if(has_next)
            {
                *reinterpret_cast<u16x8*>(
                    b_lds + next_buffer * b_buffer_elements
                        + b_row * stride_b + b_half) = staged_b;
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
            }
        }

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
