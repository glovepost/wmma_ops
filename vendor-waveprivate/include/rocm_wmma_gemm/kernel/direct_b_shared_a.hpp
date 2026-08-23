/*
 * Direct-B / shared-A block-prepacked experiment for gfx1151.
 *
 * A is cooperatively staged in a 24 KiB two-slot ring.  B fragments are read
 * directly from the persistent block/K-major input; the four M waves for each
 * N band intentionally request the same cache lines.  Refilling the inactive
 * A slot while the current tile computes removes the overwrite barrier, so a
 * single workgroup barrier publishes each K16 step while retaining two blocks
 * per CU.  Two B fragment slots overlap each cache load with four independent
 * WMMA instructions.
 */
#ifndef ROCM_WMMA_GEMM_DIRECT_B_SHARED_A_HPP
#define ROCM_WMMA_GEMM_DIRECT_B_SHARED_A_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_DBSA_PADDING_A
#define WMMA_DBSA_PADDING_A 8
#endif
#ifndef WMMA_DBSA_SWIZZLE
#define WMMA_DBSA_SWIZZLE 16
#endif

struct direct_b_shared_a_gemm
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
        constexpr int stride_a = block_k + WMMA_DBSA_PADDING_A;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        constexpr int a_buffer_elements = block_m * stride_a;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        __shared__ half a_lds[2 * a_buffer_elements];

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
                    WMMA_DBSA_SWIZZLE>()
            .map_tile(static_cast<int>(blockIdx.x),
                      grid_m,
                      grid_n,
                      &block_row,
                      &block_col);

        const int k_tiles = K / block_k;
        const int block_m_index = block_row / block_m;
        const int block_n_index = block_col / block_n;
        const half* const a_tile = A
            + static_cast<size_t>(block_m_index) * k_tiles * a_tile_elements;
        const half* const b_tile = B
            + static_cast<size_t>(block_n_index) * k_tiles * b_tile_elements;

        u16x8 staged_a0 = reinterpret_cast<const u16x8*>(a_tile)[2 * tid];
        u16x8 staged_a1 = reinterpret_cast<const u16x8*>(a_tile)[2 * tid + 1];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = staged_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = staged_a1;
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

            fragment<half, wmma_tile> a_frag[4];
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const half* source = a_lds
                    + current_buffer * a_buffer_elements
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, stride_a, block_k);
            }

            fragment<half, wmma_tile> b_frag[2];
            const half* current_b = b_tile + k_tile * b_tile_elements;
            auto load_b = [&]<int slot>(int wn)
            {
                const half* source = current_b
                    + (warp_n_base + wn * wmma_tile + half_lane) * block_k;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag[slot], source, block_k, block_k);
            };

            load_b.template operator()<0>(0);
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                if(wn + 1 < 4)
                {
                    if((wn & 1) == 0)
                        load_b.template operator()<1>(wn + 1);
                    else
                        load_b.template operator()<0>(wn + 1);
                }
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");

                // The direct-B waits would serialize an earlier A prefetch.
                // Launch it only after the final B wait, then cover its VMEM
                // latency with the last four independent WMMAs.
                if(has_next && wn == 3)
                {
                    const u16x8* next_a = reinterpret_cast<const u16x8*>(
                        a_tile + (k_tile + 1) * a_tile_elements);
                    staged_a0 = next_a[2 * tid];
                    staged_a1 = next_a[2 * tid + 1];
                }

                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if(wn < 2)
                        wmma<false>(
                            a_frag[wm], b_frag[wn & 1], accum[wm][wn]);
                    else
                        wmma<true>(
                            a_frag[wm], b_frag[wn & 1], accum[wm][wn - 2]);
                }
            }

            if(has_next)
            {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                *reinterpret_cast<u16x8*>(
                    a_lds + next_buffer * a_buffer_elements
                        + tid * stride_a) = staged_a0;
                *reinterpret_cast<u16x8*>(
                    a_lds + next_buffer * a_buffer_elements
                        + tid * stride_a + 8) = staged_a1;
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
