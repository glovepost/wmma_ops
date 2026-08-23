/*
 * B-resident, A-streamed block/K-major GEMM experiment for gfx1151.
 *
 * The retained p8 kernel holds four A fragments and streams one B fragment,
 * exposing three serial B LDS load/wait gaps per K16.  This schedule reverses
 * that choice: all four B fragments stay resident while two A fragments form
 * a small in-wave pipeline.  The LDS traffic, p8 layout, workgroup geometry,
 * and two-barrier handoff remain unchanged.  On the last A row, each B
 * fragment dies after one WMMA, giving the global prefetches registers to
 * reuse without extending their live range through the compute cluster.
 */
#ifndef ROCM_WMMA_GEMM_B_RESIDENT_HPP
#define ROCM_WMMA_GEMM_B_RESIDENT_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct b_resident_gemm
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
        const int b_row = tid >> 1;
        const int b_half = (tid & 1) * 8;

        u16x8 next_a0
            = reinterpret_cast<const u16x8*>(a_tile)[2 * tid];
        u16x8 next_a1
            = reinterpret_cast<const u16x8*>(a_tile)[2 * tid + 1];
        u16x8 next_b = reinterpret_cast<const u16x8*>(b_tile)[tid];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
        *reinterpret_cast<u16x8*>(
            b_lds + b_row * stride_b + b_half) = next_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];

        auto compute_step = [&]<bool do_prefetch>(const half* next_a,
                                                   const half* next_b_ptr)
        {
            fragment<half, wmma_tile> b_frag[4];
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                const half* source = b_lds
                    + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag[wn], source, block_k, stride_b);
            }

            fragment<half, wmma_tile> a_frag[2];
            #pragma unroll
            for(int wm = 0; wm < 2; ++wm)
            {
                const half* source = a_lds
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, stride_a, block_k);
            }

            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const int slot = wm & 1;
                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    if(wn < 2)
                        wmma<false>(
                            a_frag[slot], b_frag[wn], accum[wm][wn]);
                    else
                        wmma<true>(
                            a_frag[slot], b_frag[wn], accum[wm][wn - 2]);

                    if constexpr(do_prefetch)
                    {
                        if(wm == 3)
                        {
                            const u16x8* a_vectors
                                = reinterpret_cast<const u16x8*>(next_a);
                            const u16x8* b_vectors
                                = reinterpret_cast<const u16x8*>(next_b_ptr);
                            if(wn == 0)
                                next_a0 = a_vectors[2 * tid];
                            else if(wn == 1)
                                next_a1 = a_vectors[2 * tid + 1];
                            else if(wn == 2)
                                next_b = b_vectors[tid];
                        }
                    }
                }

                if(wm < 2)
                {
                    const half* source = a_lds
                        + (warp_m_base + (wm + 2) * wmma_tile + half_lane)
                            * stride_a;
                    load_matrix<m_input::matrix_a, m_layout::row_major>(
                        a_frag[slot], source, stride_a, block_k);
                }
            }
        };

        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
        {
            compute_step.template operator()<true>(
                a_tile + (k_tile + 1) * a_tile_elements,
                b_tile + (k_tile + 1) * b_tile_elements);
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
        compute_step.template operator()<false>(nullptr, nullptr);

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
