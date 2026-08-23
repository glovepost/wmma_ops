/*
 * Pair-local A handoff experiment for gfx1151.
 *
 * The retained 256x128 block has two N waves for every 64-row M stripe.  Both
 * waves consume the same A stripe, so they can acknowledge its final LDS read
 * without waiting for the other six waves.  Each pair then replaces only its
 * own 64 rows while B is filled into an inactive block-wide slot.  One ordinary
 * workgroup barrier publishes both operands for the next K16 step.
 *
 * This preserves the persistent block/K-major input contract and the proven
 * p8 LDS banking while removing the first of the baseline's two barriers.  A
 * remains single-buffered (12 KiB), B double-buffered (12 KiB), and every
 * global input vector is still fetched exactly once per block and K tile.
 */
#ifndef ROCM_WMMA_GEMM_PAIRED_A_HANDOFF_HPP
#define ROCM_WMMA_GEMM_PAIRED_A_HANDOFF_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_PAH_SWIZZLE
#define WMMA_PAH_SWIZZLE 16
#endif
#ifndef WMMA_PAH_CYCLIC_FINAL_REFILL
#define WMMA_PAH_CYCLIC_FINAL_REFILL 0
#endif
#ifndef WMMA_PAH_CYCLIC_PREFETCH_ONLY
#define WMMA_PAH_CYCLIC_PREFETCH_ONLY 0
#endif
struct paired_a_handoff_gemm
{
    using as3_volatile_int_ptr
        = volatile int __attribute__((address_space(3)))*;

    static __device__ __forceinline__ void wave_store(
        as3_volatile_int_ptr value, int desired, int lane)
    {
        if(lane == 0)
            *value = desired;
    }

    static __device__ __forceinline__ void wave_wait_equal(
        as3_volatile_int_ptr value, int expected)
    {
        // Poll one wave-uniform LDS address and reduce the broadcast result to
        // an SGPR.  Expressing this as a C volatile loop makes LLVM maintain a
        // per-lane exit mask even though every lane observes the same value.
        const uint32_t address = static_cast<uint32_t>(
            reinterpret_cast<uintptr_t>(value));
        uint32_t vector_value;
        uint32_t scalar_value;
        asm volatile(
            "1:\n\t"
            "ds_load_b32 %0, %2\n\t"
            "s_waitcnt lgkmcnt(0)\n\t"
            "v_readfirstlane_b32 %1, %0\n\t"
            "s_cmp_eq_u32 %1, %3\n\t"
            "s_cbranch_scc0 1b"
            : "=&v"(vector_value), "=&s"(scalar_value)
            : "v"(address), "s"(expected)
            : "memory", "scc");
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
        constexpr int stride_a = block_k + 8;
        constexpr int stride_b = block_k + 8;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        constexpr int b_buffer_elements = block_n * stride_b;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        __shared__ half a_lds[block_m * stride_a];
        __shared__ half b_lds[2 * b_buffer_elements];
        __shared__ volatile int a_read_done[4][2];

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
                    WMMA_PAH_SWIZZLE>()
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

        if(tid < 8)
            a_read_done[tid >> 1][tid & 1] = -1;

        u16x8 next_a0 = reinterpret_cast<const u16x8*>(a_tile)[2 * tid];
        u16x8 next_a1 = reinterpret_cast<const u16x8*>(a_tile)[2 * tid + 1];
        u16x8 next_b = reinterpret_cast<const u16x8*>(b_tile)[tid];
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
        *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
        const int b_row = tid >> 1;
        const int b_half = (tid & 1) * 8;
        *reinterpret_cast<u16x8*>(b_lds + b_row * stride_b + b_half) = next_b;
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        __syncthreads();

        fragment<half, wmma_tile> accum[4][2];
        const int warp_m_base = warp_row * 4 * wmma_tile;
        const int warp_n_base = warp_col * 4 * wmma_tile;

        #pragma unroll 1
        for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int current_b_buffer = k_tile & 1;
            const int next_b_buffer = current_b_buffer ^ 1;
            // The fixed 4096 record shape has 256 K16 tiles.  Cycling the final
            // prefetch to tile zero keeps the hot loop uniform and valid, at
            // the cost of one harmless refill after the final accumulation.
            const bool has_next = k_tile + 1 < k_tiles;
            const int next_k_tile = (WMMA_PAH_CYCLIC_FINAL_REFILL
                                     || WMMA_PAH_CYCLIC_PREFETCH_ONLY)
                ? ((k_tile + 1) & (k_tiles - 1))
                : (k_tile + 1);

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
                if(WMMA_PAH_CYCLIC_FINAL_REFILL
                   || WMMA_PAH_CYCLIC_PREFETCH_ONLY || has_next)
                {
                    const u16x8* next_a_vectors
                        = reinterpret_cast<const u16x8*>(
                            a_tile + next_k_tile * a_tile_elements);
                    const u16x8* next_b_vectors
                        = reinterpret_cast<const u16x8*>(
                            b_tile + next_k_tile * b_tile_elements);
                    if(wn == 0)
                        next_a0 = next_a_vectors[2 * tid];
                    else if(wn == 1)
                        next_a1 = next_a_vectors[2 * tid + 1];
                    else if(wn == 2)
                        next_b = next_b_vectors[tid];
                }

                fragment<half, wmma_tile> b_frag;
                const half* source = b_lds
                    + current_b_buffer * b_buffer_elements
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

            if(WMMA_PAH_CYCLIC_FINAL_REFILL || has_next)
            {
                // Retire the pair's A reads before publishing its generation.
                // Both waves wait for each other; other M stripes are disjoint.
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                auto* own_flag = (as3_volatile_int_ptr)(
                    reinterpret_cast<uintptr_t>(
                        &a_read_done[warp_row][warp_col]));
                auto* peer_flag = (as3_volatile_int_ptr)(
                    reinterpret_cast<uintptr_t>(
                        &a_read_done[warp_row][warp_col ^ 1]));
                wave_store(own_flag, k_tile, lane);
                wave_wait_equal(peer_flag, k_tile);

                // VMEM does not participate in the read-done protocol.  Let
                // the next-tile loads continue while this wave waits for its
                // partner, then retire them immediately before the LDS writes.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
                *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
                *reinterpret_cast<u16x8*>(
                    b_lds + next_b_buffer * b_buffer_elements
                        + b_row * stride_b + b_half) = next_b;
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
