/* Direct-fragment GEMM for explicitly prepacked A-row/B-column inputs. */
#ifndef ROCM_WMMA_GEMM_PREPACKED_DIRECT_HPP
#define ROCM_WMMA_GEMM_PREPACKED_DIRECT_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct prepacked_direct_gemm
{
    __launch_bounds__(4 * warp_size, 2)
    static __global__ void run(
        half* __restrict__ C,
        const half* __restrict__ A,
        const half* __restrict__ B,
        int M,
        int N,
        int K)
    {
        constexpr int block_m = 128;
        constexpr int block_n = 128;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int wave = static_cast<int>(threadIdx.x) / warp_size;
        const int half_lane = lane & 15;
        const int half_wave = lane >> 4;
        const int warp_row = wave / 2;
        const int warp_col = wave % 2;
        const int warp_m_base = warp_row * 64;
        const int warp_n_base = warp_col * 64;
        const int grid_n = N / block_n;
        const int tile = static_cast<int>(blockIdx.x);
        const int block_row = (tile / grid_n) * block_m;
        const int block_col = (tile % grid_n) * block_n;

        fragment<half, wmma_tile> c_frag[2][4];

        for(int k = 0; k < K; k += wmma_tile)
        {
            fragment<half, wmma_tile> b_frag[4];
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                const half* source = B
                    + static_cast<size_t>(block_col + warp_n_base
                                          + wn * wmma_tile + half_lane) * K
                    + k;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag[wn], source, K, N);
            }

            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                fragment<half, wmma_tile> a_frag;
                const half* source = A
                    + static_cast<size_t>(block_row + warp_m_base
                                          + wm * wmma_tile + half_lane) * K
                    + k;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag, source, M, K);
                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    if(wm < 2)
                        wmma<false>(a_frag, b_frag[wn], c_frag[wm][wn]);
                    else
                        wmma<true>(a_frag, b_frag[wn], c_frag[wm - 2][wn]);
                }
            }
        }

        #pragma unroll
        for(int wm = 0; wm < 4; ++wm)
        {
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                if(wm < 2)
                    store_matrix<m_layout::row_major, false, false>(
                        C, c_frag[wm][wn],
                        block_row + warp_m_base + wm * wmma_tile + half_wave,
                        block_col + warp_n_base + wn * wmma_tile + half_lane,
                        M, N);
                else
                    store_matrix<m_layout::row_major, false, true>(
                        C, c_frag[wm - 2][wn],
                        block_row + warp_m_base + wm * wmma_tile + half_wave,
                        block_col + warp_n_base + wn * wmma_tile + half_lane,
                        M, N);
            }
        }
    }
};

} // namespace rocm_wmma_gemm

#endif
