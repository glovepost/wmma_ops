/*
 * Experimental gfx1151 inter-wave producer/consumer GEMM.
 *
 * One wave owns global-to-LDS traffic and fills a two-slot ring.  Four consumer
 * waves own disjoint 64x64 output tiles and acknowledge a slot only after all
 * of their fragment reads have retired.  LDS atomics replace block barriers so
 * the producer can prepare K+1 while the consumers execute WMMA for K.
 *
 * This deliberately specializes the record shape/layout: half inputs/output,
 * A column-major, B/C row-major, and dimensions divisible by 128/16.
 */

#ifndef ROCM_WMMA_GEMM_PRODUCER_CONSUMER_HPP
#define ROCM_WMMA_GEMM_PRODUCER_CONSUMER_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_PC_WIDE
#define WMMA_PC_WIDE 0
#endif
#ifndef WMMA_PC_NARROW
#define WMMA_PC_NARROW 0
#endif
#ifndef WMMA_PC_PRODUCERS
#define WMMA_PC_PRODUCERS 1
#endif

struct producer_consumer_gemm
{
    static __device__ __forceinline__ int wave_load(const volatile int* value, int lane)
    {
        int observed = 0;
        if(lane == 0)
            observed = *value;
        return __shfl(observed, 0, 32);
    }

    static __device__ __forceinline__ void wave_wait_at_least(
        const volatile int* value, int expected, int lane)
    {
        while(wave_load(value, lane) < expected)
            ;
        __threadfence_block();
    }

    static __device__ __forceinline__ void wave_wait_equal(
        const volatile int* value, int expected, int lane)
    {
        while(wave_load(value, lane) != expected)
            ;
        __threadfence_block();
    }

    __launch_bounds__(((WMMA_PC_WIDE ? 8 : 4) + WMMA_PC_PRODUCERS) * warp_size, 1)
    static __global__ void run(
        half* __restrict__ C,
        const half* __restrict__ A,
        const half* __restrict__ B,
        int M,
        int N,
        int K)
    {
        constexpr int block_m = 128;
        constexpr int block_n = WMMA_PC_NARROW ? 64 : 128;
        constexpr int block_k = 16;
        constexpr int consumer_waves = WMMA_PC_WIDE ? 8 : 4;
        constexpr int producer_waves = WMMA_PC_PRODUCERS;
        constexpr int warp_cols = WMMA_PC_WIDE ? 4 : 2;
        constexpr int warp_tile_n = (WMMA_PC_WIDE || WMMA_PC_NARROW) ? 2 : 4;
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));

        __shared__ half a_lds[2][block_k][block_m];
        __shared__ half b_lds[2][block_k][block_n];
        __shared__ volatile int ready[2];
        __shared__ volatile int producer_done[2];
        __shared__ volatile int consumed[2][consumer_waves];

        const int tid = static_cast<int>(threadIdx.x);
        const int wave = tid / warp_size;
        const int lane = tid & (warp_size - 1);
        const int grid_n = N / block_n;
        const int tile = static_cast<int>(blockIdx.x);
        const int block_row = (tile / grid_n) * block_m;
        const int block_col = (tile % grid_n) * block_n;

        if(tid < 2)
        {
            ready[tid] = -1;
            producer_done[tid] = 0;
        }
        if(tid < 2 * consumer_waves)
            consumed[tid / consumer_waves][tid % consumer_waves] = -1;
        __syncthreads();

        if(wave < producer_waves)
        {
            constexpr int a_vectors = block_k * block_m / 8;
            constexpr int b_vectors = block_k * block_n / 8;
            for(int k_tile = 0, generation = 0; k_tile < K;
                k_tile += block_k, ++generation)
            {
                const int slot = generation & 1;
                if(generation >= 2)
                {
                    #pragma unroll
                    for(int consumer = 0; consumer < consumer_waves; ++consumer)
                        wave_wait_equal(&consumed[slot][consumer], generation - 2, lane);
                }

                #pragma unroll 1
                for(int vector = wave * warp_size + lane;
                    vector < a_vectors + b_vectors;
                    vector += producer_waves * warp_size)
                {
                    if(vector < a_vectors)
                    {
                        const int k_local = vector / (block_m / 8);
                        const int m_vector = vector % (block_m / 8);
                        const half* source = A
                            + static_cast<size_t>(k_tile + k_local) * M
                            + block_row + m_vector * 8;
                        *reinterpret_cast<u16x8*>(
                            &a_lds[slot][k_local][m_vector * 8])
                            = *reinterpret_cast<const u16x8*>(source);
                    }
                    else
                    {
                        const int b_vector = vector - a_vectors;
                        const int k_local = b_vector / (block_n / 8);
                        const int n_vector = b_vector % (block_n / 8);
                        const half* source = B
                            + static_cast<size_t>(k_tile + k_local) * N
                            + block_col + n_vector * 8;
                        *reinterpret_cast<u16x8*>(
                            &b_lds[slot][k_local][n_vector * 8])
                            = *reinterpret_cast<const u16x8*>(source);
                    }
                }

                // Every lane waits for its VMEM and LDS writes before lane zero
                // publishes the generation to consumer waves.
                __builtin_amdgcn_s_waitcnt(0);
                __threadfence_block();
                if(lane == 0)
                {
                    const int epoch_target
                        = (generation / 2 + 1) * producer_waves;
                    const int prior = atomicAdd(
                        const_cast<int*>(&producer_done[slot]), 1);
                    if(prior + 1 == epoch_target)
                    {
                        __threadfence_block();
                        ready[slot] = generation;
                    }
                }
            }
            return;
        }

        const int consumer = wave - producer_waves;
        const int warp_row = consumer / warp_cols;
        const int warp_col = consumer % warp_cols;
        const int half_lane = lane & 15;
        const int half_wave = lane >> 4;
        const int warp_m_base = warp_row * 64;
        const int warp_n_base = warp_col * warp_tile_n * wmma_tile;

        // OPSEL packs logical M tiles 0/2 and 1/3 into low/high halves.
        fragment<half, wmma_tile> c_frag[2][warp_tile_n];

        for(int k_tile = 0, generation = 0; k_tile < K;
            k_tile += block_k, ++generation)
        {
            const int slot = generation & 1;
            wave_wait_at_least(&ready[slot], generation, lane);

            fragment<half, wmma_tile> b_frag[warp_tile_n];
            #pragma unroll
            for(int wn = 0; wn < warp_tile_n; ++wn)
            {
                const half* source = &b_lds[slot][0]
                    [warp_n_base + wn * wmma_tile + half_lane];
                load_matrix<m_input::matrix_b, m_layout::row_major>(
                    b_frag[wn], source, block_k, block_n);
            }

            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                fragment<half, wmma_tile> a_frag;
                const half* source = &a_lds[slot][0]
                    [warp_m_base + wm * wmma_tile + half_lane];
                load_matrix<m_input::matrix_a, m_layout::col_major>(
                    a_frag, source, block_m, block_k);
                #pragma unroll
                for(int wn = 0; wn < warp_tile_n; ++wn)
                {
                    if(wm < 2)
                        wmma<false>(a_frag, b_frag[wn], c_frag[wm][wn]);
                    else
                        wmma<true>(a_frag, b_frag[wn], c_frag[wm - 2][wn]);
                }
            }

            // The WMMA instructions depend on every LDS read above.  Retiring
            // LGKM before the acknowledgement makes slot reuse race-free.
            __builtin_amdgcn_s_waitcnt(0);
            if(lane == 0)
                consumed[slot][consumer] = generation;
        }

        #pragma unroll
        for(int wm = 0; wm < 4; ++wm)
        {
            #pragma unroll
            for(int wn = 0; wn < warp_tile_n; ++wn)
            {
                if(wm < 2)
                    store_matrix<m_layout::row_major, false, false>(
                        C,
                        c_frag[wm][wn],
                        block_row + warp_m_base + wm * wmma_tile + half_wave,
                        block_col + warp_n_base + wn * wmma_tile + half_lane,
                        M,
                        N);
                else
                    store_matrix<m_layout::row_major, false, true>(
                        C,
                        c_frag[wm - 2][wn],
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
