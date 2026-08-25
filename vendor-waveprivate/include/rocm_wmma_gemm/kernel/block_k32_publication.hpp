/*
 * Two-slice K32 publication stage for gfx1151.
 *
 * A 40-half row keeps both K16 slices in one p4 tile.  The complete A+B tile
 * occupies 30 KiB, so two workgroups remain resident in a 64-KiB CU LDS while
 * each read-complete/publication barrier pair covers twice as much WMMA work.
 */
#ifndef ROCM_WMMA_GEMM_BLOCK_K32_PUBLICATION_HPP
#define ROCM_WMMA_GEMM_BLOCK_K32_PUBLICATION_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

struct block_k32_publication_gemm
{
    using u16x8 = uint16_t __attribute__((ext_vector_type(8)));
    struct alignas(16) packed_halves
    {
        u16x8 low;
        u16x8 high;
    };

    static __device__ __forceinline__ u16x8 load_lds_b128(const half* source)
    {
        const uint32_t address = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(source));
        u16x8 value;
        asm volatile("ds_load_b128 %0, %1"
                     : "=v"(value)
                     : "v"(address)
                     : "memory");
        return value;
    }

    static __device__ __forceinline__ void store_lds_b128(
        half* destination, u16x8 value)
    {
        const uint32_t address = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(destination));
        asm volatile("ds_store_b128 %0, %1"
                     :
                     : "v"(address), "v"(value)
                     : "memory");
    }

    static __device__ __forceinline__ void load_fragment_b128(
        fragment<half, wmma_tile>& frag, const half* source)
    {
        packed_halves packed;
        packed.low = load_lds_b128(source);
        packed.high = load_lds_b128(source + 8);
        frag.get() = __builtin_bit_cast(
            typename fragment<half, wmma_tile>::frag_vec,
            packed);
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
        constexpr int block_k = 32;
        constexpr int stride = block_k + 8;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        constexpr int a_lds_elements = block_m * stride;
        constexpr int b_lds_elements = block_n * stride;
        __shared__ alignas(16) half lds[a_lds_elements + b_lds_elements];
        half* const a_lds = lds;
        half* const b_lds = lds + a_lds_elements;

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
        u16x8 stage3;
        u16x8 stage4;
        u16x8 stage5;

        auto prefetch_group = [&](const half* next_a,
                                  const half* next_b,
                                  int group)
        {
            const u16x8* a_vectors = reinterpret_cast<const u16x8*>(next_a);
            const u16x8* b_vectors = reinterpret_cast<const u16x8*>(next_b);
            if(group == 0)
            {
                stage0 = a_vectors[4 * tid];
                stage1 = a_vectors[4 * tid + 1];
            }
            else if(group == 1)
            {
                stage2 = a_vectors[4 * tid + 2];
                stage3 = a_vectors[4 * tid + 3];
            }
            else
            {
                stage4 = b_vectors[2 * tid];
                stage5 = b_vectors[2 * tid + 1];
            }
        };

        auto commit_group = [&](int group)
        {
            if(group == 0)
            {
                store_lds_b128(a_lds + tid * stride, stage0);
                store_lds_b128(a_lds + tid * stride + 8, stage1);
            }
            else if(group == 1)
            {
                store_lds_b128(a_lds + tid * stride + 16, stage2);
                store_lds_b128(a_lds + tid * stride + 24, stage3);
            }
            else
            {
                const int b_row = tid >> 1;
                const int b_offset = (tid & 1) * 16;
                store_lds_b128(b_lds + b_row * stride + b_offset, stage4);
                store_lds_b128(
                    b_lds + b_row * stride + b_offset + 8, stage5);
            }
        };

        prefetch_group(a_tile, b_tile, 0);
        prefetch_group(a_tile, b_tile, 1);
        prefetch_group(a_tile, b_tile, 2);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        commit_group(0);
        commit_group(1);
        commit_group(2);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];

        auto compute_slice = [&]<int slice, bool do_prefetch>(
                                 const half* next_a,
                                 const half* next_b)
        {
            fragment<half, wmma_tile> a_frag[4];
            auto load_a = [&]<int wm>()
            {
                const half* source = a_lds
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride
                    + slice * wmma_tile;
                load_fragment_b128(a_frag[wm], source);
            };

            load_a.template operator()<0>();
            load_a.template operator()<1>();
            __builtin_amdgcn_sched_barrier(0);
            fragment<half, wmma_tile> b0;
            load_fragment_b128(
                b0,
                b_lds
                    + (warp_n_base + half_lane) * stride + slice * wmma_tile);
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
                __builtin_amdgcn_sched_barrier(0);
                fragment<half, wmma_tile> b_frag;
                load_fragment_b128(
                    b_frag,
                    b_lds
                        + (warp_n_base + wn * wmma_tile + half_lane) * stride
                        + slice * wmma_tile);
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
                            // Each final A-fragment use releases eight VGPRs,
                            // exactly one pair of next-tile b128 vectors.
                            __builtin_amdgcn_sched_barrier(0);
                            prefetch_group(next_a, next_b, wm);
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
            }
        };

        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
        {
            const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
            const half* next_b = b_tile + (k_tile + 1) * b_tile_elements;
            compute_slice.template operator()<0, false>(nullptr, nullptr);
            compute_slice.template operator()<1, true>(next_a, next_b);

            // Both resident slices are now consumed.  Publish the next K32
            // tile with one read-complete and one publication barrier.
            __syncthreads();
            asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
            commit_group(0);
            asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
            commit_group(1);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            commit_group(2);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            __syncthreads();
        }
        compute_slice.template operator()<0, false>(nullptr, nullptr);
        compute_slice.template operator()<1, false>(nullptr, nullptr);

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
