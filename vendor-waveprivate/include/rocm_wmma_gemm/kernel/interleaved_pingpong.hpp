/*
 * Compact ping-pong GEMM experiment for gfx1151.
 *
 * The retained p8 single buffer has good LDS bank behavior but two barriers
 * per K16 step.  Two ordinary p8 buffers consume 36 KiB and lose the second
 * resident workgroup.  This layout interleaves the two buffers within a
 * 40-half row pitch:
 *
 *   buffer 0: [0, 16), buffer 1: [24, 40)
 *
 * A row-to-row displacement is therefore 80 bytes (20 banks), the modular
 * inverse of p8's 48-byte (12-bank) displacement.  Both operands fit in
 * 30 KiB, retain two-workgroup LDS residency, and need only the publish
 * barrier after filling the inactive buffer.
 */
#ifndef ROCM_WMMA_GEMM_INTERLEAVED_PINGPONG_HPP
#define ROCM_WMMA_GEMM_INTERLEAVED_PINGPONG_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct interleaved_pingpong_gemm
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
        constexpr int row_pitch = 40;
        constexpr int buffer_1_offset = 24;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        __shared__ half a_lds[block_m * row_pitch];
        __shared__ half b_lds[block_n * row_pitch];

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
        *reinterpret_cast<u16x8*>(a_lds + tid * row_pitch) = next_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * row_pitch + 8) = next_a1;
        *reinterpret_cast<u16x8*>(
            b_lds + b_row * row_pitch + b_half) = next_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];

        auto compute_step = [&]<int current_buffer,
                                int next_buffer,
                                bool do_prefetch>(
                                    const half* next_a,
                                    const half* next_b_ptr)
        {
            constexpr int current_offset
                = current_buffer ? buffer_1_offset : 0;
            constexpr int next_offset = next_buffer ? buffer_1_offset : 0;
            fragment<half, wmma_tile> a_frag[4];
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const half* source = a_lds
                    + (warp_m_base + wm * wmma_tile + half_lane) * row_pitch
                    + current_offset;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, row_pitch, block_k);
            }

            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                if constexpr(do_prefetch)
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

                fragment<half, wmma_tile> b_frag;
                const half* source = b_lds
                    + (warp_n_base + wn * wmma_tile + half_lane) * row_pitch
                    + current_offset;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag, source, block_k, row_pitch);
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

            if constexpr(do_prefetch)
            {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                *reinterpret_cast<u16x8*>(
                    a_lds + tid * row_pitch + next_offset) = next_a0;
                *reinterpret_cast<u16x8*>(
                    a_lds + tid * row_pitch + next_offset + 8) = next_a1;
                *reinterpret_cast<u16x8*>(
                    b_lds + b_row * row_pitch + next_offset + b_half)
                    = next_b;
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
            }
        };

        int k_tile = 0;
        #pragma unroll 1
        for(; k_tile + 2 < k_tiles; k_tile += 2)
        {
            compute_step.template operator()<0, 1, true>(
                a_tile + (k_tile + 1) * a_tile_elements,
                b_tile + (k_tile + 1) * b_tile_elements);
            compute_step.template operator()<1, 0, true>(
                a_tile + (k_tile + 2) * a_tile_elements,
                b_tile + (k_tile + 2) * b_tile_elements);
        }
        compute_step.template operator()<0, 1, true>(
            a_tile + (k_tile + 1) * a_tile_elements,
            b_tile + (k_tile + 1) * b_tile_elements);
        compute_step.template operator()<1, 0, false>(nullptr, nullptr);

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
