/*
 * Padding-free half-row-XOR ping-pong GEMM for gfx1151.
 *
 * Each 16-byte half-row is placed at one of eight LDS bank phases by XORing
 * its physical half with row bit 2.  Unlike a padded row pitch, this bijection
 * consumes no extra storage: two complete 256x16 A and 128x16 B stages fit in
 * 24 KiB.  Compute reads one stage while cooperative refills write the other,
 * reducing publication to one workgroup barrier per K16.
 */
#ifndef ROCM_WMMA_GEMM_COMPACT_XOR_PINGPONG_HPP
#define ROCM_WMMA_GEMM_COMPACT_XOR_PINGPONG_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct compact_xor_pingpong_gemm
{
    using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

    static __device__ __forceinline__ void load_fragment(
        fragment<half, wmma_tile>& frag,
        const half* first,
        const half* second)
    {
        using as3_u16x8_ptr
            = const u16x8 __attribute__((address_space(3)))*;
        const auto* first_vector = (as3_u16x8_ptr)(
            reinterpret_cast<uintptr_t>(first));
        const auto* second_vector = (as3_u16x8_ptr)(
            reinterpret_cast<uintptr_t>(second));
        auto* packed = reinterpret_cast<u16x8*>(&frag.get());
        packed[0] = *first_vector;
        packed[1] = *second_vector;
    }

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
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        __shared__ alignas(16) half a_lds[2 * a_tile_elements];
        __shared__ alignas(16) half b_lds[2 * b_tile_elements];

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

        u16x8 next_a0;
        u16x8 next_a1;
        u16x8 next_b;

        auto prefetch = [&](const half* next_a, const half* next_b_ptr, int part)
        {
            const u16x8* a_vectors = reinterpret_cast<const u16x8*>(next_a);
            const u16x8* b_vectors
                = reinterpret_cast<const u16x8*>(next_b_ptr);
            if(part == 0)
                next_a0 = a_vectors[2 * tid];
            else if(part == 1)
                next_a1 = a_vectors[2 * tid + 1];
            else if(part == 2)
                next_b = b_vectors[tid];
        };

        auto commit = [&](int buffer)
        {
            const int a_buffer = buffer * a_tile_elements;
            const int a_half_xor = ((tid >> 2) & 1) * 8;
            *reinterpret_cast<u16x8*>(
                a_lds + a_buffer + tid * block_k + a_half_xor) = next_a0;
            *reinterpret_cast<u16x8*>(
                a_lds + a_buffer + tid * block_k + (a_half_xor ^ 8)) = next_a1;

            const int b_buffer = buffer * b_tile_elements;
            const int b_row = tid >> 1;
            const int b_half = (tid & 1) * 8;
            const int b_physical_half = b_half ^ (((b_row >> 2) & 1) * 8);
            *reinterpret_cast<u16x8*>(
                b_lds + b_buffer + b_row * block_k + b_physical_half) = next_b;
        };

        prefetch(a_tile, b_tile, 0);
        prefetch(a_tile, b_tile, 1);
        prefetch(a_tile, b_tile, 2);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        commit(0);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];

        auto compute_step = [&]<int current_buffer,
                                int next_buffer,
                                bool do_prefetch>(
                                    const half* next_a,
                                    const half* next_b_ptr)
        {
            constexpr int a_current = current_buffer * a_tile_elements;
            constexpr int b_current = current_buffer * b_tile_elements;
            fragment<half, wmma_tile> a_frag[4];
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const int row
                    = warp_m_base + wm * wmma_tile + half_lane;
                const int half_xor = ((row >> 2) & 1) * 8;
                const half* row_base
                    = a_lds + a_current + row * block_k;
                load_fragment(
                    a_frag[wm], row_base + half_xor,
                    row_base + (half_xor ^ 8));
            }

            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                if constexpr(do_prefetch)
                    prefetch(next_a, next_b_ptr, wn);

                const int row
                    = warp_n_base + wn * wmma_tile + half_lane;
                const int half_xor = ((row >> 2) & 1) * 8;
                const half* row_base
                    = b_lds + b_current + row * block_k;
                fragment<half, wmma_tile> b_frag;
                load_fragment(
                    b_frag, row_base + half_xor,
                    row_base + (half_xor ^ 8));
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
                commit(next_buffer);
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
            }
        };

        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles; k_tile += 2)
        {
            compute_step.template operator()<0, 1, true>(
                a_tile + (k_tile + 1) * a_tile_elements,
                b_tile + (k_tile + 1) * b_tile_elements);
            // The final refill wraps to a valid tile and is deliberately
            // discarded.  Keeping both phases structurally identical avoids
            // LLVM's separately allocated 184-VGPR tail; its one-time cost is
            // amortized over the 256 K16 slices of the record shape.
            const int following = k_tile + 2 < k_tiles ? k_tile + 2 : 0;
            compute_step.template operator()<1, 0, true>(
                a_tile + following * a_tile_elements,
                b_tile + following * b_tile_elements);
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
