/*
 * Wide-M block/K-major GEMM experiment for gfx1151.
 *
 * A 512x128 workgroup contains sixteen ordinary 64x64 consumer waves.  It
 * doubles the retained leader's output and A tile while reusing each staged B
 * tile across eight wave rows.  P8 staging occupies 30 KiB, so two workgroups
 * still fit in a 64-KiB CU LDS budget.  Relative to two 256x128 workgroups it
 * halves workgroup-barrier arrivals per output element without changing the
 * per-wave accumulator footprint.  Both eight-wave halves cooperatively load
 * the same B vectors; the duplicate, cache-resident loads and identical LDS
 * stores avoid a divergent loader path that inflated allocation above the
 * two-block VGPR threshold.
 */
#ifndef ROCM_WMMA_GEMM_BLOCK_512X128_HPP
#define ROCM_WMMA_GEMM_BLOCK_512X128_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct block_512x128_gemm
{
    __global__ __launch_bounds__(512)
        __attribute__((amdgpu_waves_per_eu(2, 8))) static void run(
            half* __restrict__ C,
            const half* __restrict__ A,
            const half* __restrict__ B,
            int M,
            int N,
            int K)
    {
        constexpr int block_m = 512;
        constexpr int block_n = 128;
        constexpr int block_k = wmma_tile;
        constexpr int stride_a = block_k + 8;
        constexpr int stride_b = block_k + 8;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        __shared__ half a_lds[block_m * stride_a];
        __shared__ half b_lds[block_n * stride_b];

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
                    16>()
            .map_tile(static_cast<int>(blockIdx.x),
                      grid_m,
                      grid_n,
                      &block_row,
                      &block_col);

        const int block_m_index = block_row / block_m;
        const int block_n_index = block_col / block_n;
        const int k_tiles = K / block_k;
        const half* const a_tile = A
            + static_cast<size_t>(block_m_index) * k_tiles * a_tile_elements;
        const half* const b_tile = B
            + static_cast<size_t>(block_n_index) * k_tiles * b_tile_elements;
        const int warp_m_base = warp_row * 4 * wmma_tile;
        const int warp_n_base = warp_col * 4 * wmma_tile;
        const int b_loader = tid & 255;
        const int b_row = b_loader >> 1;
        const int b_half = (b_loader & 1) * 8;

        u16x8 next_a0
            = reinterpret_cast<const u16x8*>(a_tile)[2 * tid];
        u16x8 next_a1
            = reinterpret_cast<const u16x8*>(a_tile)[2 * tid + 1];
        u16x8 next_b
            = reinterpret_cast<const u16x8*>(b_tile)[b_loader];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
        *reinterpret_cast<u16x8*>(
            b_lds + b_row * stride_b + b_half) = next_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];
        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
        {
            const half* const next_a = a_tile
                + (k_tile + 1) * a_tile_elements;
            const half* const next_b_ptr = b_tile
                + (k_tile + 1) * b_tile_elements;
            fragment<half, wmma_tile> a_frag[4];
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const half* source = a_lds
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, stride_a, block_k);
            }

            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                const u16x8* a_vectors
                    = reinterpret_cast<const u16x8*>(next_a);
                if(wn == 0)
                    next_a0 = a_vectors[2 * tid];
                else if(wn == 1)
                    next_a1 = a_vectors[2 * tid + 1];
                else if(wn == 2)
                    next_b
                        = reinterpret_cast<const u16x8*>(next_b_ptr)[b_loader];

                fragment<half, wmma_tile> b_frag;
                const half* source = b_lds
                    + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag, source, block_k, stride_b);
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if(wn < 2)
                        wmma<false>(a_frag[wm], b_frag, accum[wm][wn]);
                    else
                        wmma<true>(
                            a_frag[wm], b_frag, accum[wm][wn - 2]);
                }
            }

            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();
            *reinterpret_cast<u16x8*>(
                a_lds + tid * stride_a) = next_a0;
            *reinterpret_cast<u16x8*>(
                a_lds + tid * stride_a + 8) = next_a1;
            *reinterpret_cast<u16x8*>(
                b_lds + b_row * stride_b + b_half) = next_b;
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            __syncthreads();
        }

        fragment<half, wmma_tile> a_frag[4];
        #pragma unroll
        for(int wm = 0; wm < 4; ++wm)
        {
            const half* source = a_lds
                + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
            load_matrix<m_input::matrix_a, m_layout::row_major>(
                a_frag[wm], source, stride_a, block_k);
        }
        #pragma unroll
        for(int wn = 0; wn < 4; ++wn)
        {
            fragment<half, wmma_tile> b_frag;
            const half* source = b_lds
                + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
            load_matrix<m_input::matrix_b, m_layout::col_major>(
                b_frag, source, block_k, stride_b);
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                if(wn < 2)
                    wmma<false>(a_frag[wm], b_frag, accum[wm][wn]);
                else
                    wmma<true>(a_frag[wm], b_frag, accum[wm][wn - 2]);
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
