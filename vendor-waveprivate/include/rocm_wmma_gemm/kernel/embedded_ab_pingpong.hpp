/*
 * Embedded-padding A/B ping-pong GEMM for gfx1151.
 *
 * The selected p8 tile reserves eight unused half elements after every
 * 16-half A row. Across 256 A rows those padding cells hold exactly the 256
 * b128 half-rows of the 128x16 B tile. A modular permutation preserves B's
 * original p8 LDS bank phase while embedding it in A's padding. Two complete
 * A+B tiles therefore occupy 24 KiB instead of 36 KiB, retaining two-block
 * CU residency and allowing one publication barrier per K16 slice.
 */
#ifndef ROCM_WMMA_GEMM_EMBEDDED_AB_PINGPONG_HPP
#define ROCM_WMMA_GEMM_EMBEDDED_AB_PINGPONG_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct embedded_ab_pingpong_gemm
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
        constexpr int stride = block_k + 8;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        constexpr int lds_tile_elements = block_m * stride;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        struct alignas(16) packed_halves
        {
            u16x8 low;
            u16x8 high;
        };
        static_assert(sizeof(packed_halves)
                          == sizeof(typename fragment<half, wmma_tile>::frag_vec),
                      "embedded B halves must exactly fill one WMMA fragment");

        __shared__ half tiles[2 * lds_tile_elements];

        const int tid = static_cast<int>(threadIdx.x);
        const int lane = tid & (warp_size - 1);
        const int wave = tid / warp_size;
        const int half_lane = lane & 15;
        const int half_wave = lane >> 4;
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
        const half* a_tile = A
            + static_cast<size_t>(block_m_index) * k_tiles * a_tile_elements;
        const half* b_tile = B
            + static_cast<size_t>(block_n_index) * k_tiles * b_tile_elements;
        const int warp_m_base = warp_row * 4 * wmma_tile;
        const int warp_n_base = warp_col * 4 * wmma_tile;

        u16x8 stage0;
        u16x8 stage1;
        u16x8 stage2;

        auto prefetch_stage = [&](const half* next_a,
                                  const half* next_b,
                                  int stage)
        {
            if(stage == 0)
                stage0 = reinterpret_cast<const u16x8*>(next_a)[2 * tid];
            else if(stage == 1)
                stage1 = reinterpret_cast<const u16x8*>(next_a)[2 * tid + 1];
            else
                stage2 = reinterpret_cast<const u16x8*>(next_b)[tid];
        };

        auto b_padding_slot = [](int row, int half_index)
        {
            // Address modulo the 128-byte bank period:
            //   (slot*48 + 32) == (row*48 + half_index*16) mod 128.
            // The quotient term assigns all 256 (row, half) pairs uniquely.
            const int residue = (row + 2 + 3 * half_index) & 7;
            const int quotient = 2 * (row >> 3) + half_index;
            return residue + 8 * quotient;
        };

        auto commit_stages = [&](int buffer)
        {
            half* tile = tiles + buffer * lds_tile_elements;
            *reinterpret_cast<u16x8*>(tile + tid * stride) = stage0;
            *reinterpret_cast<u16x8*>(tile + tid * stride + 8) = stage1;

            const int b_row = tid >> 1;
            const int b_half = tid & 1;
            const int slot = b_padding_slot(b_row, b_half);
            *reinterpret_cast<u16x8*>(tile + slot * stride + block_k) = stage2;
        };

        // The embedding repeats its residue every 16 logical B rows:
        // slot(row + 16, half) = slot(row, half) + 32.  All four WMMA B
        // fragments can therefore share two lane-dependent bases and use a
        // compile-time 1,536-byte step, rather than rebuilding eight LDS
        // addresses from the active ping-pong buffer on every K slice.
        const int first_b_row = warp_n_base + half_lane;
        const int b_low_base_element =
            b_padding_slot(first_b_row, 0) * stride + block_k;
        const int b_high_base_element =
            b_padding_slot(first_b_row, 1) * stride + block_k;
        constexpr int b_fragment_step = 32 * stride;

        auto load_embedded_b = [&](fragment<half, wmma_tile>& frag,
                                   int buffer,
                                   int wn)
        {
            const half* tile = tiles + buffer * lds_tile_elements;
            packed_halves packed;
            packed.low = *reinterpret_cast<const u16x8*>(
                tile + b_low_base_element + wn * b_fragment_step);
            packed.high = *reinterpret_cast<const u16x8*>(
                tile + b_high_base_element + wn * b_fragment_step);
            frag.get() = __builtin_bit_cast(
                typename fragment<half, wmma_tile>::frag_vec, packed);
        };

        prefetch_stage(a_tile, b_tile, 0);
        prefetch_stage(a_tile, b_tile, 1);
        prefetch_stage(a_tile, b_tile, 2);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        commit_stages(0);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];

        auto compute = [&]<bool do_prefetch>(int buffer,
                                              const half* next_a,
                                              const half* next_b)
        {
            const half* tile = tiles + buffer * lds_tile_elements;
            fragment<half, wmma_tile> a_frag[4];
            auto load_a = [&]<int wm>()
            {
                const half* source = tile
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, stride, block_k);
            };

            // Match the selected critical path: A0/A1, B0, then A2/A3. B0 is
            // therefore ready at lgkmcnt(4), while the younger A fragments
            // retire progressively behind the first WMMA pair.
            load_a.template operator()<0>();
            load_a.template operator()<1>();
            __builtin_amdgcn_sched_barrier(0);
            fragment<half, wmma_tile> b0;
            load_embedded_b(b0, buffer, 0);
            __builtin_amdgcn_sched_barrier(0);
            load_a.template operator()<2>();
            load_a.template operator()<3>();
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            wmma<false>(a_frag[0], b0, accum[0][0]);
            wmma<false>(a_frag[1], b0, accum[1][0]);
            asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
            wmma<false>(a_frag[2], b0, accum[2][0]);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            wmma<false>(a_frag[3], b0, accum[3][0]);
            __builtin_amdgcn_sched_barrier(0);

            #pragma unroll
            for(int wn = 1; wn < 4; ++wn)
            {
                // Keep one B fragment live at a time. Without these compiler
                // scheduling fences LLVM hoists all four noncontiguous pairs,
                // expanding the image from the intended 8-register B window
                // to 32 registers and crossing the two-block VGPR boundary.
                __builtin_amdgcn_sched_barrier(0);
                fragment<half, wmma_tile> b_frag;
                load_embedded_b(b_frag, buffer, wn);
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if(wn == 1)
                        wmma<false>(a_frag[wm], b_frag, accum[wm][wn]);
                    else
                        wmma<true>(a_frag[wm], b_frag, accum[wm][wn - 2]);

                    if constexpr(do_prefetch)
                    {
                        if(wn == 3 && wm < 3)
                        {
                            // Refill only after this A fragment's final WMMA;
                            // the scheduling fence makes its dead registers
                            // available to the corresponding stage value.
                            __builtin_amdgcn_sched_barrier(0);
                            prefetch_stage(next_a, next_b, wm);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        };

        int current_buffer = 0;
        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
        {
            const int next_buffer = 1 - current_buffer;
            const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
            const half* next_b = b_tile + (k_tile + 1) * b_tile_elements;
            compute.template operator()<true>(current_buffer, next_a, next_b);

            // The next tile is disjoint, so no read-complete barrier precedes
            // these stores. Progressive retirement retains VMEM overlap.
            asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
            half* next_tile = tiles + next_buffer * lds_tile_elements;
            *reinterpret_cast<u16x8*>(next_tile + tid * stride) = stage0;
            asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
            *reinterpret_cast<u16x8*>(next_tile + tid * stride + 8) = stage1;
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            const int b_row = tid >> 1;
            const int b_half = tid & 1;
            const int slot = b_padding_slot(b_row, b_half);
            *reinterpret_cast<u16x8*>(
                next_tile + slot * stride + block_k) = stage2;
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            __syncthreads();
            current_buffer = next_buffer;
        }
        compute.template operator()<false>(current_buffer, nullptr, nullptr);

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
