/*
 * Experimental K32 / two-slot K16 ring for persistent prepacked inputs.
 *
 * Input contract:
 *   A[bm][bk32][slice][m_local][k16]
 *   B[bn][bk32][slice][n_local][k16]
 *
 * The two K16 slices share one padded K32 LDS tile.  A slice is consumed while
 * the replacement for that slot is fetched into 12 VGPRs; the following
 * barrier publishes that refill after all waves have retired the old slot.
 */
#ifndef ROCM_WMMA_GEMM_BLOCK_K2_RING_HPP
#define ROCM_WMMA_GEMM_BLOCK_K2_RING_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_K2_RING_PADDING_A
#define WMMA_K2_RING_PADDING_A 8
#endif
#ifndef WMMA_K2_RING_PADDING_B
#define WMMA_K2_RING_PADDING_B 8
#endif
#ifndef WMMA_K2_RING_SWIZZLE
#define WMMA_K2_RING_SWIZZLE 16
#endif

struct block_k2_ring_gemm
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
        constexpr int block_k = 32;
        constexpr int stride_a = block_k + WMMA_K2_RING_PADDING_A;
        constexpr int stride_b = block_k + WMMA_K2_RING_PADDING_B;
        constexpr int a_slice_elements = block_m * wmma_tile;
        constexpr int b_slice_elements = block_n * wmma_tile;
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
                    WMMA_K2_RING_SWIZZLE>()
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

        u16x8 next_a0;
        u16x8 next_a1;
        u16x8 next_b;

        // Seed both ring slots.  Each slice is contiguous in global memory.
        const u16x8* a_vectors = reinterpret_cast<const u16x8*>(a_tile);
        const u16x8* b_vectors = reinterpret_cast<const u16x8*>(b_tile);
        next_a0 = a_vectors[2 * tid];
        next_a1 = a_vectors[2 * tid + 1];
        next_b = b_vectors[tid];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
        const int b_row = tid >> 1;
        const int b_half = (tid & 1) * 8;
        *reinterpret_cast<u16x8*>(b_lds + b_row * stride_b + b_half) = next_b;

        a_vectors = reinterpret_cast<const u16x8*>(a_tile + a_slice_elements);
        b_vectors = reinterpret_cast<const u16x8*>(b_tile + b_slice_elements);
        next_a0 = a_vectors[2 * tid];
        next_a1 = a_vectors[2 * tid + 1];
        next_b = b_vectors[tid];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + wmma_tile)
            = next_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + wmma_tile + 8)
            = next_a1;
        *reinterpret_cast<u16x8*>(
            b_lds + b_row * stride_b + wmma_tile + b_half) = next_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];
        const int warp_m_base = warp_row * 4 * wmma_tile;
        const int warp_n_base = warp_col * 4 * wmma_tile;
        const int total_slices = K / wmma_tile;

        #pragma unroll 1
        for(int slice = 0; slice < total_slices; ++slice)
        {
            const int slot_offset = (slice & 1) * wmma_tile;
            const bool has_next = slice + 2 < total_slices;
            const half* next_a_slice = nullptr;
            const half* next_b_slice = nullptr;
            if(has_next)
            {
                const int next_slice = slice + 2;
                const int next_tile = next_slice >> 1;
                const int next_slot = next_slice & 1;
                next_a_slice = a_tile + next_tile * a_tile_elements
                    + next_slot * a_slice_elements;
                next_b_slice = b_tile + next_tile * b_tile_elements
                    + next_slot * b_slice_elements;
            }

            fragment<half, wmma_tile> a_frag[4];
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const half* source = a_lds
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride_a
                    + slot_offset;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, stride_a, block_k);
            }

            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
                if(has_next)
                {
                    const u16x8* next_a_vectors
                        = reinterpret_cast<const u16x8*>(next_a_slice);
                    const u16x8* next_b_vectors
                        = reinterpret_cast<const u16x8*>(next_b_slice);
                    if(wn == 0)
                        next_a0 = next_a_vectors[2 * tid];
                    else if(wn == 1)
                        next_a1 = next_a_vectors[2 * tid + 1];
                    else if(wn == 2)
                        next_b = next_b_vectors[tid];
                }

                fragment<half, wmma_tile> b_frag;
                const half* source = b_lds
                    + (warp_n_base + wn * wmma_tile + half_lane) * stride_b
                    + slot_offset;
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
                // Keep the following B LDS read behind this WMMA group. LLVM
                // otherwise hoists all four B fragments and extends 24 VGPRs
                // across the complete N step on gfx1151.
                asm volatile("" ::: "memory");
            }

            if(has_next)
            {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
                *reinterpret_cast<u16x8*>(
                    a_lds + tid * stride_a + slot_offset) = next_a0;
                *reinterpret_cast<u16x8*>(
                    a_lds + tid * stride_a + slot_offset + 8) = next_a1;
                *reinterpret_cast<u16x8*>(
                    b_lds + b_row * stride_b + slot_offset + b_half) = next_b;
            }
            else if(slice == total_slices - 2)
            {
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
