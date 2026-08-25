/*
 * Experimental persistent-input GEMM for gfx1151.
 *
 * A and B are packed into block/K-major 16-wide microtiles before timing:
 *   A[bm][bk][m_local][k_local]
 *   B[bn][bk][n_local][k_local]
 * This preserves cooperative, contiguous global reads while placing both LDS
 * operands in the native WMMA fragment order.  It is a different input
 * contract from the ordinary column-major-A/row-major-B benchmark.
 */
#ifndef ROCM_WMMA_GEMM_BLOCK_PREPACKED_HPP
#define ROCM_WMMA_GEMM_BLOCK_PREPACKED_HPP

#include "kernel.hpp"

namespace rocm_wmma_gemm
{

#ifndef WMMA_BP_PADDING_A
#define WMMA_BP_PADDING_A 2
#endif
#ifndef WMMA_BP_PADDING_B
#define WMMA_BP_PADDING_B 2
#endif
#ifndef WMMA_BP_PACK_N
#define WMMA_BP_PACK_N 0
#endif
#ifndef WMMA_BP_DOUBLE_BUFFER
#define WMMA_BP_DOUBLE_BUFFER 0
#endif
#ifndef WMMA_BP_DOUBLE_BUFFER_STATIC
#define WMMA_BP_DOUBLE_BUFFER_STATIC 0
#endif
#ifndef WMMA_BP_DOUBLE_BUFFER_LATE
#define WMMA_BP_DOUBLE_BUFFER_LATE 0
#endif
#ifndef WMMA_BP_BLOCK_N
#define WMMA_BP_BLOCK_N 128
#endif
#ifndef WMMA_BP_BLOCK_M
#define WMMA_BP_BLOCK_M 256
#endif
#ifndef WMMA_BP_SWIZZLE
#define WMMA_BP_SWIZZLE 16
#endif
#ifndef WMMA_BP_WAVES_MIN
#define WMMA_BP_WAVES_MIN 2
#endif
#ifndef WMMA_BP_WAVES_MAX
#define WMMA_BP_WAVES_MAX 8
#endif
#ifndef WMMA_BP_SPLIT_BARRIER
#define WMMA_BP_SPLIT_BARRIER 0
#endif
#ifndef WMMA_BP_WAIT_AFTER_BARRIER
#define WMMA_BP_WAIT_AFTER_BARRIER 0
#endif
#ifndef WMMA_BP_NO_EXPLICIT_VMWAIT
#define WMMA_BP_NO_EXPLICIT_VMWAIT 0
#endif
#ifndef WMMA_BP_B_PAIR
#define WMMA_BP_B_PAIR 0
#endif
#ifndef WMMA_BP_K_SLICES
#define WMMA_BP_K_SLICES 1
#endif
#ifndef WMMA_BP_K2_RING
#define WMMA_BP_K2_RING 0
#endif
#ifndef WMMA_BP_PREFETCH_A0_WN
#define WMMA_BP_PREFETCH_A0_WN 0
#endif
#ifndef WMMA_BP_PREFETCH_A1_WN
#define WMMA_BP_PREFETCH_A1_WN 1
#endif
#ifndef WMMA_BP_PREFETCH_B_WN
#define WMMA_BP_PREFETCH_B_WN 2
#endif
#ifndef WMMA_BP_HALF_SWIZZLE
#define WMMA_BP_HALF_SWIZZLE 0
#endif
#ifndef WMMA_BP_WARP_TILE_M
#define WMMA_BP_WARP_TILE_M 4
#endif
#ifndef WMMA_BP_WARP_TILE2_REPAIR
#define WMMA_BP_WARP_TILE2_REPAIR 0
#endif
#ifndef WMMA_BP_SET_PRIO
#define WMMA_BP_SET_PRIO 0
#endif
#ifndef WMMA_BP_VECTOR_EPILOGUE
#define WMMA_BP_VECTOR_EPILOGUE 0
#endif
#ifndef WMMA_BP_FULL_TILE_STORE
#define WMMA_BP_FULL_TILE_STORE 0
#endif
#ifndef WMMA_BP_STREAM_B_BARRIER
#define WMMA_BP_STREAM_B_BARRIER 0
#endif
#ifndef WMMA_BP_LATE_B1_PREFETCH
#define WMMA_BP_LATE_B1_PREFETCH 0
#endif
#ifndef WMMA_BP_LATE_B_REFILL
#define WMMA_BP_LATE_B_REFILL 0
#endif
#ifndef WMMA_BP_BUFFER_A_PREFETCH
#define WMMA_BP_BUFFER_A_PREFETCH 0
#endif
#ifndef WMMA_BP_HYBRID_A_PINGPONG
#define WMMA_BP_HYBRID_A_PINGPONG 0
#endif
#ifndef WMMA_BP_HYBRID_A_GROUP_LOADS
#define WMMA_BP_HYBRID_A_GROUP_LOADS 0
#endif
#ifndef WMMA_BP_HYBRID_A_LATE_COMMIT
#define WMMA_BP_HYBRID_A_LATE_COMMIT 0
#endif
#ifndef WMMA_BP_HYBRID_B_PINGPONG
#define WMMA_BP_HYBRID_B_PINGPONG 0
#endif

struct block_prepacked_gemm
{
    __global__ __launch_bounds__(
        (WMMA_BP_BLOCK_M / (WMMA_BP_WARP_TILE_M * wmma_tile))
        * (WMMA_BP_BLOCK_N / (4 * wmma_tile)) * warp_size)
        __attribute__((amdgpu_waves_per_eu(WMMA_BP_WAVES_MIN, WMMA_BP_WAVES_MAX))) static void run(
            half* __restrict__ C,
            const half* __restrict__ A,
            const half* __restrict__ B,
            int M,
            int N,
            int K)
    {
        constexpr int block_m = WMMA_BP_BLOCK_M;
        constexpr int block_n = WMMA_BP_BLOCK_N;
        constexpr int warp_tile_m = WMMA_BP_WARP_TILE_M;
        constexpr int block_k = WMMA_BP_K_SLICES * wmma_tile;
        constexpr int stride_a = block_k + WMMA_BP_PADDING_A;
        constexpr int stride_b = block_k + WMMA_BP_PADDING_B;
        constexpr int a_tile_elements = block_m * block_k;
        constexpr int b_tile_elements = block_n * block_k;
        constexpr int lds_buffers = WMMA_BP_DOUBLE_BUFFER ? 2 : 1;
        constexpr int a_lds_buffers
            = WMMA_BP_HYBRID_A_PINGPONG ? 2 : lds_buffers;
        constexpr int b_lds_buffers
            = WMMA_BP_HYBRID_B_PINGPONG ? 2 : lds_buffers;
        constexpr int warp_cols = block_n / (4 * wmma_tile);
        static_assert((block_m == 128 || block_m == 256)
                      && (block_n == 128 || block_n == 256));
        static_assert(WMMA_BP_K_SLICES == 1 || WMMA_BP_K_SLICES == 2);
        static_assert(!WMMA_BP_WARP_TILE2_REPAIR
                          || (WMMA_BP_PACK_N && block_m == 128
                              && block_n == 128 && warp_tile_m == 2
                              && WMMA_BP_K_SLICES == 1
                              && !WMMA_BP_DOUBLE_BUFFER),
                      "warp-tile-2 repair specializes N-packed 128x128 K16 single buffering");
        static_assert(!WMMA_BP_WARP_TILE2_REPAIR
                          || (WMMA_BP_PREFETCH_A0_WN >= 0
                              && WMMA_BP_PREFETCH_A0_WN < 4
                              && WMMA_BP_PREFETCH_B_WN >= 0
                              && WMMA_BP_PREFETCH_B_WN < 4),
                      "warp-tile-2 prefetch positions must name a WMMA N step");
        static_assert(!WMMA_BP_LATE_B_REFILL
                          || (WMMA_BP_PACK_N && block_m == 128
                              && block_n == 128 && !WMMA_BP_DOUBLE_BUFFER
                              && warp_tile_m == 4 && !WMMA_BP_HALF_SWIZZLE
                              && !WMMA_BP_LATE_B1_PREFETCH),
                      "late B refill specializes N-packed 128x128 single buffering");
        static_assert(!WMMA_BP_K2_RING
                          || (WMMA_BP_K_SLICES == 2 && WMMA_BP_PACK_N
                              && block_m == 256 && block_n == 128
                              && !WMMA_BP_DOUBLE_BUFFER),
                      "K2 ring specializes N-packed 256x128 single buffering");
        static_assert(!WMMA_BP_HYBRID_A_PINGPONG
                          || (WMMA_BP_PACK_N && block_m == 256
                              && block_n == 128 && !WMMA_BP_DOUBLE_BUFFER
                              && WMMA_BP_K_SLICES == 1
                              && warp_tile_m == 4 && !WMMA_BP_HALF_SWIZZLE),
                      "hybrid A ping-pong specializes N-packed 256x128 K16");
        static_assert(!WMMA_BP_HYBRID_B_PINGPONG
                          || (WMMA_BP_PACK_N && block_m == 256
                              && block_n == 128 && !WMMA_BP_DOUBLE_BUFFER
                              && !WMMA_BP_HYBRID_A_PINGPONG
                              && WMMA_BP_K_SLICES == 1
                              && warp_tile_m == 4 && !WMMA_BP_HALF_SWIZZLE),
                      "hybrid B ping-pong specializes N-packed 256x128 K16");
        using u16x8 = uint16_t __attribute__((ext_vector_type(8)));
        using i32x4 = int32_t __attribute__((ext_vector_type(4)));

        __shared__ half a_lds[a_lds_buffers * block_m * stride_a];
        __shared__ half b_lds[b_lds_buffers * block_n * stride_b];

        const int tid = static_cast<int>(threadIdx.x);
        const int lane = tid & (warp_size - 1);
        const int wave = tid / warp_size;
        const int half_lane = lane & 15;
        const int half_wave = lane >> 4;
        const int warp_row = wave / warp_cols;
        const int warp_col = wave % warp_cols;

        const int grid_m = M / block_m;
        const int grid_n = N / block_n;
        int block_row = 0;
        int block_col = 0;
        tile_mapper<block_m,
                    block_n,
                    m_layout::col_major,
                    m_layout::row_major,
                    WMMA_BP_SWIZZLE>()
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

        const int warp_m_base = warp_row * warp_tile_m * wmma_tile;
        const int warp_n_base = warp_col * 4 * wmma_tile;

        u16x8 next_a0;
        u16x8 next_a1;
        u16x8 next_b;
        u16x8 next_b1;
        u16x8 next_a_k2[4];
        u16x8 next_b_k2[2];

        auto prefetch_all = [&](const half* next_a, const half* next_b_ptr)
        {
            const u16x8* a_vectors = reinterpret_cast<const u16x8*>(next_a);
            const u16x8* b_vectors = reinterpret_cast<const u16x8*>(next_b_ptr);
            if constexpr(WMMA_BP_WARP_TILE2_REPAIR)
            {
                // The compact 128x128 block has 256 vectors in each input
                // tile and exactly 256 threads.  Give every thread one A and
                // one B vector; the former warp/lane scheme covered only half
                // of B and the generic block_m==128 path then overwrote both
                // registers with out-of-bounds 2*tid loads.
                next_a0 = a_vectors[tid];
                next_b = b_vectors[tid];
            }
            else if constexpr(block_n == 128)
            {
                if constexpr(warp_tile_m == 2)
                {
                    next_a0 = a_vectors[tid];
                    if(lane < 16)
                        next_b = b_vectors[wave * 16 + lane];
                }
                else
                {
                    next_a0 = a_vectors[2 * tid];
                    next_a1 = a_vectors[2 * tid + 1];
                }
            }
            else
                next_a0 = a_vectors[tid];
            if constexpr(!WMMA_BP_WARP_TILE2_REPAIR && block_m == 128)
            {
                next_b = b_vectors[2 * tid];
                next_b1 = b_vectors[2 * tid + 1];
            }
            else if constexpr(!WMMA_BP_WARP_TILE2_REPAIR && warp_tile_m != 2)
                next_b = b_vectors[tid];
        };

        auto commit_all = [&]()
        {
            if constexpr(block_n == 128)
            {
                if constexpr(warp_tile_m == 2)
                {
                    const int a_row = tid >> 1;
                    const int a_half = (tid & 1) * 8;
                    *reinterpret_cast<u16x8*>(
                        a_lds + a_row * stride_a + a_half) = next_a0;
                }
                else if constexpr(WMMA_BP_HALF_SWIZZLE)
                {
                    const int half_xor = ((tid >> 2) & 1) * 8;
                    *reinterpret_cast<u16x8*>(
                        a_lds + tid * stride_a + half_xor) = next_a0;
                    *reinterpret_cast<u16x8*>(
                        a_lds + tid * stride_a + (half_xor ^ 8)) = next_a1;
                }
                else
                {
                    *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
                    *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
                }
            }
            else
            {
                const int a_row = tid >> 1;
                const int a_half = (tid & 1) * 8;
                *reinterpret_cast<u16x8*>(a_lds + a_row * stride_a + a_half)
                    = next_a0;
            }
            if constexpr(warp_tile_m == 2)
            {
                if constexpr(WMMA_BP_WARP_TILE2_REPAIR)
                {
                    const int b_row = tid >> 1;
                    const int b_half = (tid & 1) * 8;
                    *reinterpret_cast<u16x8*>(
                        b_lds + b_row * stride_b + b_half) = next_b;
                }
                else if(lane < 16)
                {
                    const int vector = wave * 16 + lane;
                    const int row = vector >> 1;
                    const int row_half = (vector & 1) * 8;
                    *reinterpret_cast<u16x8*>(
                        b_lds + row * stride_b + row_half) = next_b;
                }
            }
            else
            {
                const int b_row = tid >> 1;
                const int b_half = (tid & 1) * 8;
                if constexpr(block_m == 128)
                {
                    *reinterpret_cast<u16x8*>(b_lds + tid * stride_b) = next_b;
                    *reinterpret_cast<u16x8*>(b_lds + tid * stride_b + 8) = next_b1;
                }
                else
                {
                    const int physical_half = WMMA_BP_HALF_SWIZZLE
                        ? (b_half ^ (((b_row >> 2) & 1) * 8))
                        : b_half;
                    *reinterpret_cast<u16x8*>(
                        b_lds + b_row * stride_b + physical_half) = next_b;
                }
            }
        };

        // A live-range split for the four-wave 128x128 experiment. Keep the A
        // refill overlapped with WMMA, commit it after the handoff barrier,
        // then load and immediately commit B. This deliberately sacrifices B
        // load/compute overlap so A and B refill vectors need not be live at
        // the same time. It is opt-in because the resource/latency tradeoff
        // must be measured on gfx1151 rather than inferred from occupancy.
        auto commit_a_128 = [&]()
        {
            *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
            *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
        };
        auto refill_and_commit_b_128 = [&](const half* next_b_ptr)
        {
            const u16x8* b_vectors
                = reinterpret_cast<const u16x8*>(next_b_ptr);
            const u16x8 late_b0 = b_vectors[2 * tid];
            const u16x8 late_b1 = b_vectors[2 * tid + 1];
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            *reinterpret_cast<u16x8*>(b_lds + tid * stride_b) = late_b0;
            *reinterpret_cast<u16x8*>(b_lds + tid * stride_b + 8) = late_b1;
        };
        auto commit_b_256 = [&]()
        {
            const int b_row = tid >> 1;
            const int b_half = (tid & 1) * 8;
            *reinterpret_cast<u16x8*>(
                b_lds + b_row * stride_b + b_half) = next_b;
        };
        auto commit_a_256 = [&]()
        {
            *reinterpret_cast<u16x8*>(a_lds + tid * stride_a) = next_a0;
            *reinterpret_cast<u16x8*>(a_lds + tid * stride_a + 8) = next_a1;
        };
        auto buffer_load_a_128 = [&](const half* next_a,
                                     int vector_byte_offset,
                                     int scalar_byte_offset)
        {
            const auto resource = __builtin_amdgcn_make_buffer_rsrc(
                const_cast<half*>(next_a), 0, -1, 0x31004000);
            const i32x4 words = __builtin_amdgcn_raw_buffer_load_b128(
                resource, vector_byte_offset, scalar_byte_offset, 0);
            return __builtin_bit_cast(u16x8, words);
        };

        auto prefetch_k2 = [&](const half* next_a, const half* next_b_ptr)
        {
            static_assert(WMMA_BP_K_SLICES == 1
                          || (block_m == 256 && block_n == 128));
            const u16x8* a_vectors = reinterpret_cast<const u16x8*>(next_a);
            const u16x8* b_vectors = reinterpret_cast<const u16x8*>(next_b_ptr);
            #pragma unroll
            for(int vector = 0; vector < 4; ++vector)
                next_a_k2[vector] = a_vectors[4 * tid + vector];
            #pragma unroll
            for(int vector = 0; vector < 2; ++vector)
                next_b_k2[vector] = b_vectors[2 * tid + vector];
        };

        auto commit_k2 = [&]()
        {
            #pragma unroll
            for(int vector = 0; vector < 4; ++vector)
            {
                const int flat = 4 * tid + vector;
                const int row = flat >> 2;
                const int row_vector = flat & 3;
                *reinterpret_cast<u16x8*>(
                    a_lds + row * stride_a + row_vector * 8) = next_a_k2[vector];
            }
            #pragma unroll
            for(int vector = 0; vector < 2; ++vector)
            {
                const int flat = 2 * tid + vector;
                const int row = flat >> 2;
                const int row_vector = flat & 3;
                *reinterpret_cast<u16x8*>(
                    b_lds + row * stride_b + row_vector * 8) = next_b_k2[vector];
            }
        };

        // The K2 ring keeps two K16 slices in one padded K32 LDS tile. The
        // persistent input contract is [block][K32][slice][row][K16], so each
        // refill remains one contiguous, coalesced transaction per operand.
        auto prefetch_ring_slice = [&](const half* next_a,
                                       const half* next_b_ptr)
        {
            const u16x8* a_vectors = reinterpret_cast<const u16x8*>(next_a);
            const u16x8* b_vectors
                = reinterpret_cast<const u16x8*>(next_b_ptr);
            next_a0 = a_vectors[2 * tid];
            next_a1 = a_vectors[2 * tid + 1];
            next_b = b_vectors[tid];
        };

        auto commit_ring_slice = [&](int slot)
        {
            const int slot_offset = slot * wmma_tile;
            *reinterpret_cast<u16x8*>(
                a_lds + tid * stride_a + slot_offset) = next_a0;
            *reinterpret_cast<u16x8*>(
                a_lds + tid * stride_a + slot_offset + 8) = next_a1;
            const int b_row = tid >> 1;
            const int b_half = (tid & 1) * 8;
            *reinterpret_cast<u16x8*>(
                b_lds + b_row * stride_b + slot_offset + b_half) = next_b;
        };

        if constexpr(WMMA_BP_K2_RING)
        {
            constexpr int a_slice_elements = block_m * wmma_tile;
            constexpr int b_slice_elements = block_n * wmma_tile;
            prefetch_ring_slice(a_tile, b_tile);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            commit_ring_slice(0);
            prefetch_ring_slice(a_tile + a_slice_elements,
                                b_tile + b_slice_elements);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            commit_ring_slice(1);
        }
        else if constexpr(WMMA_BP_K_SLICES == 1)
            prefetch_all(a_tile, b_tile);
        else
            prefetch_k2(a_tile, b_tile);
        if constexpr(!WMMA_BP_K2_RING)
        {
            __builtin_amdgcn_s_waitcnt(0);
            if constexpr(WMMA_BP_K_SLICES == 1)
                commit_all();
            else
                commit_k2();
        }
        __builtin_amdgcn_s_waitcnt(0x7f);
        __syncthreads();

        fragment<half, wmma_tile> c_m[2][4];
        fragment<half, wmma_tile> c_n[4][2];

        auto load_half_swizzled = [&](fragment<half, wmma_tile>& frag,
                                      const half* first_source,
                                      const half* second_source)
        {
            using as3_u16x8_ptr
                = const u16x8 __attribute__((address_space(3)))*;
            const auto* first = (as3_u16x8_ptr)(reinterpret_cast<uintptr_t>(
                first_source));
            const auto* second = (as3_u16x8_ptr)(reinterpret_cast<uintptr_t>(
                second_source));
            auto* packed = reinterpret_cast<u16x8*>(&frag.get());
            packed[0] = *first;
            packed[1] = *second;
        };

        auto compute_tile = [&]<bool do_prefetch>(
                                const half* next_a, const half* next_b_ptr)
        {
            if constexpr(!WMMA_BP_PACK_N)
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

                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if constexpr(do_prefetch)
                    {
                        const u16x8* a_vectors
                            = reinterpret_cast<const u16x8*>(next_a);
                        const u16x8* b_vectors
                            = reinterpret_cast<const u16x8*>(next_b_ptr);
                        if constexpr(block_m == 256 && block_n == 128)
                        {
                            if(wm == 0)
                                next_a0 = a_vectors[2 * tid];
                            else if(wm == 1)
                                next_a1 = a_vectors[2 * tid + 1];
                            else if(wm == 2)
                                next_b = b_vectors[tid];
                        }
                        else if constexpr(block_m == 256 && block_n == 256)
                        {
                            if(wm == 0)
                                next_a0 = a_vectors[tid];
                            else if(wm == 1)
                                next_b = b_vectors[tid];
                        }
                        else if constexpr(block_m == 128 && block_n == 256)
                        {
                            if(wm == 0)
                                next_a0 = a_vectors[tid];
                            else if(wm == 1)
                                next_b = b_vectors[2 * tid];
                            else if(wm == 2)
                                next_b1 = b_vectors[2 * tid + 1];
                        }
                        else if constexpr(block_m == 128 && block_n == 128)
                        {
                            if(wm == 0)
                                next_a0 = a_vectors[2 * tid];
                            else if(wm == 1)
                                next_a1 = a_vectors[2 * tid + 1];
                            else if(wm == 2)
                                next_b = b_vectors[2 * tid];
                            else if(wm == 3)
                                next_b1 = b_vectors[2 * tid + 1];
                        }
                    }

                    fragment<half, wmma_tile> a_frag;
                    const half* source = a_lds
                        + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                    load_matrix<m_input::matrix_a, m_layout::row_major>(
                        a_frag, source, stride_a, block_k);

                    #pragma unroll
                    for(int wn = 0; wn < 4; ++wn)
                    {
                        if(wm < 2)
                            wmma<false>(a_frag, b_frag[wn], c_m[wm][wn]);
                        else
                            wmma<true>(a_frag, b_frag[wn], c_m[wm - 2][wn]);
                    }
                }
            }
            else
            {
                const int half_xor = ((half_lane >> 2) & 1) * 8;
                const half* a_first_base = a_lds
                    + (warp_m_base + half_lane) * stride_a + half_xor;
                const half* a_second_base = a_lds
                    + (warp_m_base + half_lane) * stride_a + (half_xor ^ 8);
                const half* b_first_base = b_lds
                    + (warp_n_base + half_lane) * stride_b + half_xor;
                const half* b_second_base = b_lds
                    + (warp_n_base + half_lane) * stride_b + (half_xor ^ 8);
                fragment<half, wmma_tile> a_frag[4];
                #pragma unroll
                for(int wm = 0; wm < warp_tile_m; ++wm)
                {
                    const half* source = a_lds
                        + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                    if constexpr(WMMA_BP_HALF_SWIZZLE)
                        load_half_swizzled(
                            a_frag[wm],
                            a_first_base + wm * wmma_tile * stride_a,
                            a_second_base + wm * wmma_tile * stride_a);
                    else
                        load_matrix<m_input::matrix_a, m_layout::row_major>(
                            a_frag[wm], source, stride_a, block_k);
                }

                if constexpr(WMMA_BP_B_PAIR)
                {
                    #pragma unroll
                    for(int pair = 0; pair < 2; ++pair)
                    {
                        fragment<half, wmma_tile> b_frag[2];
                        #pragma unroll
                        for(int local_n = 0; local_n < 2; ++local_n)
                        {
                            const int wn = pair * 2 + local_n;
                            const half* source = b_lds
                                + (warp_n_base + wn * wmma_tile + half_lane)
                                      * stride_b;
                            load_matrix<m_input::matrix_b, m_layout::col_major>(
                                b_frag[local_n], source, block_k, stride_b);
                        }

                        if constexpr(do_prefetch)
                        {
                            const u16x8* a_vectors
                                = reinterpret_cast<const u16x8*>(next_a);
                            const u16x8* b_vectors
                                = reinterpret_cast<const u16x8*>(next_b_ptr);
                            if(pair == 0)
                            {
                                next_a0 = a_vectors[2 * tid];
                                next_a1 = a_vectors[2 * tid + 1];
                            }
                            else
                                next_b = b_vectors[tid];
                        }

                        #pragma unroll
                        for(int local_n = 0; local_n < 2; ++local_n)
                        {
                            const int wn = pair * 2 + local_n;
                            #pragma unroll
                            for(int wm = 0; wm < 4; ++wm)
                            {
                                if(wn < 2)
                                    wmma<false>(
                                        a_frag[wm], b_frag[local_n], c_n[wm][wn]);
                                else
                                    wmma<true>(
                                        a_frag[wm], b_frag[local_n], c_n[wm][wn - 2]);
                            }
                        }
                    }
                }
                else
                {
                    #pragma unroll
                    for(int wn = 0; wn < 4; ++wn)
                    {
                        if constexpr(do_prefetch)
                        {
                            const u16x8* a_vectors
                                = reinterpret_cast<const u16x8*>(next_a);
                            const u16x8* b_vectors
                                = reinterpret_cast<const u16x8*>(next_b_ptr);
                            if constexpr(block_m == 256 && block_n == 128)
                            {
                                if constexpr(warp_tile_m == 2)
                                {
                                    if(wn == 0)
                                        next_a0 = a_vectors[tid];
                                    if(wn == 1 && lane < 16)
                                        next_b = b_vectors[wave * 16 + lane];
                                }
                                else
                                {
                                    if(wn == WMMA_BP_PREFETCH_A0_WN)
                                        next_a0 = a_vectors[2 * tid];
                                    if(wn == WMMA_BP_PREFETCH_A1_WN)
                                        next_a1 = a_vectors[2 * tid + 1];
                                    if(wn == WMMA_BP_PREFETCH_B_WN)
                                        next_b = b_vectors[tid];
                                }
                            }
                            else if constexpr(block_m == 256 && block_n == 256)
                            {
                                if(wn == 0)
                                    next_a0 = a_vectors[tid];
                                else if(wn == 1)
                                    next_b = b_vectors[tid];
                            }
                            else if constexpr(block_m == 128 && block_n == 256)
                            {
                                if(wn == 0)
                                    next_a0 = a_vectors[tid];
                                else if(wn == 1)
                                    next_b = b_vectors[2 * tid];
                                else if(wn == 2)
                                    next_b1 = b_vectors[2 * tid + 1];
                            }
                            else if constexpr(block_m == 128 && block_n == 128)
                            {
                                if constexpr(WMMA_BP_WARP_TILE2_REPAIR)
                                {
                                    if(wn == WMMA_BP_PREFETCH_A0_WN)
                                        next_a0 = a_vectors[tid];
                                    if(wn == WMMA_BP_PREFETCH_B_WN)
                                        next_b = b_vectors[tid];
                                }
                                else
                                {
                                    if(wn == 0)
                                    {
                                        if constexpr(WMMA_BP_BUFFER_A_PREFETCH)
                                            next_a0 = buffer_load_a_128(
                                                next_a, 32 * tid, 0);
                                        else
                                            next_a0 = a_vectors[2 * tid];
                                    }
                                    else if(wn == 1)
                                    {
                                        if constexpr(WMMA_BP_BUFFER_A_PREFETCH)
                                            next_a1 = buffer_load_a_128(
                                                next_a, 32 * tid, 16);
                                        else
                                            next_a1 = a_vectors[2 * tid + 1];
                                    }
                                    else if(wn == 2 && !WMMA_BP_LATE_B_REFILL)
                                        next_b = b_vectors[2 * tid];
                                    else if(wn == 3 && !WMMA_BP_LATE_B1_PREFETCH
                                            && !WMMA_BP_LATE_B_REFILL)
                                        next_b1 = b_vectors[2 * tid + 1];
                                }
                            }
                        }

                        fragment<half, wmma_tile> b_frag;
                        const half* source = b_lds
                            + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                        if constexpr(WMMA_BP_HALF_SWIZZLE)
                            load_half_swizzled(
                                b_frag,
                                b_first_base + wn * wmma_tile * stride_b,
                                b_second_base + wn * wmma_tile * stride_b);
                        else
                            load_matrix<m_input::matrix_b, m_layout::col_major>(
                                b_frag, source, block_k, stride_b);

                        #pragma unroll
                        for(int wm = 0; wm < warp_tile_m; ++wm)
                        {
                            if(wn < 2)
                                wmma<false>(a_frag[wm], b_frag, c_n[wm][wn]);
                            else
                                wmma<true>(a_frag[wm], b_frag, c_n[wm][wn - 2]);
                        }
                        if constexpr(WMMA_BP_STREAM_B_BARRIER)
                            __builtin_amdgcn_sched_barrier(0);
                        if constexpr(do_prefetch && WMMA_BP_LATE_B1_PREFETCH
                                     && block_m == 128 && block_n == 128)
                        {
                            if(wn == 3)
                            {
                                const u16x8* b_vectors
                                    = reinterpret_cast<const u16x8*>(next_b_ptr);
                                next_b1 = b_vectors[2 * tid + 1];
                            }
                        }
                    }
                }
            }
        };

        // Hybrid ping-pong keeps the smaller B tile single-buffered and
        // double-buffers only A.  For p8, an A tile is 12 KiB, so its buffer
        // displacement is bank-phase neutral and two complete 30-KiB
        // workgroups still fit in the 64-KiB CU LDS budget.  All current A
        // fragments are loaded before the B/WMMA loop; the next A tile can
        // therefore be stored to the inactive buffer after wn=1, overlapping
        // those two stores with the remaining eight WMMAs.  B retains the
        // original overwrite barrier and is the only operand committed in the
        // serial handoff region.
        auto compute_hybrid_a = [&]<bool do_prefetch>(
                                     int current_a_buffer,
                                     int next_a_buffer,
                                     const half* next_a,
                                     const half* next_b_ptr)
        {
            static_assert(!WMMA_BP_HYBRID_A_PINGPONG
                              || (WMMA_BP_PACK_N && block_m == 256
                                  && block_n == 128 && warp_tile_m == 4),
                          "hybrid A compute specializes the retained geometry");
            fragment<half, wmma_tile> a_frag[4];
            #pragma unroll
            for(int wm = 0; wm < 4; ++wm)
            {
                const half* source = a_lds
                    + current_a_buffer * block_m * stride_a
                    + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                load_matrix<m_input::matrix_a, m_layout::row_major>(
                    a_frag[wm], source, stride_a, block_k);
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
                    {
                        next_a0 = a_vectors[2 * tid];
                        if constexpr(WMMA_BP_HYBRID_A_GROUP_LOADS)
                            next_a1 = a_vectors[2 * tid + 1];
                    }
                    else if(wn == 1 && !WMMA_BP_HYBRID_A_GROUP_LOADS)
                        next_a1 = a_vectors[2 * tid + 1];
                    if(wn == (WMMA_BP_HYBRID_A_LATE_COMMIT ? 1 : 2))
                        next_b = b_vectors[tid];
                }

                fragment<half, wmma_tile> b_frag;
                const half* source = b_lds
                    + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag, source, block_k, stride_b);
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if(wn < 2)
                        wmma<false>(a_frag[wm], b_frag, c_n[wm][wn]);
                    else
                        wmma<true>(a_frag[wm], b_frag, c_n[wm][wn - 2]);
                }
                __builtin_amdgcn_sched_barrier(0);

                if constexpr(do_prefetch)
                {
                    if(wn == (WMMA_BP_HYBRID_A_LATE_COMMIT ? 2 : 1))
                    {
                        // In the ordinary form only the two A operations have
                        // issued. The late form issues B third, then vmcnt(1)
                        // retires both older A loads without waiting for B.
                        if constexpr(WMMA_BP_HYBRID_A_LATE_COMMIT)
                            asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
                        else
                            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                        half* next_a_lds = a_lds
                            + next_a_buffer * block_m * stride_a;
                        *reinterpret_cast<u16x8*>(
                            next_a_lds + tid * stride_a) = next_a0;
                        *reinterpret_cast<u16x8*>(
                            next_a_lds + tid * stride_a + 8) = next_a1;
                    }
                }
            }
        };

        // The asymmetric sibling double-buffers only B.  B is issued before
        // both A refills; vmcnt(2) therefore waits for that oldest operation
        // while leaving the two younger A loads in flight.  Its single LDS
        // store targets the inactive B buffer and overlaps the latter half of
        // the WMMA cluster.  Only A remains in the serial overwrite handoff.
        auto compute_hybrid_b = [&]<bool do_prefetch>(
                                     int current_b_buffer,
                                     int next_b_buffer,
                                     const half* next_a,
                                     const half* next_b_ptr)
        {
            static_assert(!WMMA_BP_HYBRID_B_PINGPONG
                              || (WMMA_BP_PACK_N && block_m == 256
                                  && block_n == 128 && warp_tile_m == 4),
                          "hybrid B compute specializes the retained geometry");
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
                if constexpr(do_prefetch)
                {
                    if(wn == 0)
                    {
                        const u16x8* b_vectors
                            = reinterpret_cast<const u16x8*>(next_b_ptr);
                        const u16x8* a_vectors
                            = reinterpret_cast<const u16x8*>(next_a);
                        next_b = b_vectors[tid];
                        next_a0 = a_vectors[2 * tid];
                        next_a1 = a_vectors[2 * tid + 1];
                    }
                }

                fragment<half, wmma_tile> b_frag;
                const half* source = b_lds
                    + current_b_buffer * block_n * stride_b
                    + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                load_matrix<m_input::matrix_b, m_layout::col_major>(
                    b_frag, source, block_k, stride_b);
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if(wn < 2)
                        wmma<false>(a_frag[wm], b_frag, c_n[wm][wn]);
                    else
                        wmma<true>(a_frag[wm], b_frag, c_n[wm][wn - 2]);
                }
                __builtin_amdgcn_sched_barrier(0);

                if constexpr(do_prefetch)
                {
                    if(wn == 1)
                    {
                        // B was issued before the two A loads, so allowing two
                        // VMEM operations to remain guarantees B is available.
                        asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
                        const int b_row = tid >> 1;
                        const int b_half = (tid & 1) * 8;
                        *reinterpret_cast<u16x8*>(
                            b_lds + next_b_buffer * block_n * stride_b
                                + b_row * stride_b + b_half) = next_b;
                    }
                }
            }
        };

        auto compute_k2 = [&]<bool do_prefetch>(
                              const half* next_a, const half* next_b_ptr)
        {
            static_assert(WMMA_BP_K_SLICES == 1
                              || (WMMA_BP_PACK_N && block_m == 256
                                  && block_n == 128 && !WMMA_BP_DOUBLE_BUFFER),
                          "K2 currently specializes the N-packed 256x128 single buffer");
            #pragma unroll
            for(int slice = 0; slice < 2; ++slice)
            {
                fragment<half, wmma_tile> a_frag[4];
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    const half* source = a_lds
                        + (warp_m_base + wm * wmma_tile + half_lane) * stride_a
                        + slice * wmma_tile;
                    load_matrix<m_input::matrix_a, m_layout::row_major>(
                        a_frag[wm], source, stride_a, block_k);
                }

                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    if constexpr(do_prefetch)
                    {
                        const int step = slice * 4 + wn;
                        const u16x8* a_vectors
                            = reinterpret_cast<const u16x8*>(next_a);
                        const u16x8* b_vectors
                            = reinterpret_cast<const u16x8*>(next_b_ptr);
                        if(step < 4)
                            next_a_k2[step] = a_vectors[4 * tid + step];
                        else if(step < 6)
                            next_b_k2[step - 4] = b_vectors[2 * tid + step - 4];
                    }

                    fragment<half, wmma_tile> b_frag;
                    const half* source = b_lds
                        + (warp_n_base + wn * wmma_tile + half_lane) * stride_b
                        + slice * wmma_tile;
                    load_matrix<m_input::matrix_b, m_layout::col_major>(
                        b_frag, source, block_k, stride_b);

                    #pragma unroll
                    for(int wm = 0; wm < 4; ++wm)
                    {
                        if(wn < 2)
                            wmma<false>(a_frag[wm], b_frag, c_n[wm][wn]);
                        else
                            wmma<true>(a_frag[wm], b_frag, c_n[wm][wn - 2]);
                    }
                }
            }
        };

        auto compute_ring = [&]<bool do_prefetch>(
                                int slot,
                                const half* next_a,
                                const half* next_b_ptr)
        {
            const int slot_offset = slot * wmma_tile;
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
                if constexpr(do_prefetch)
                {
                    const u16x8* a_vectors
                        = reinterpret_cast<const u16x8*>(next_a);
                    const u16x8* b_vectors
                        = reinterpret_cast<const u16x8*>(next_b_ptr);
                    if(wn == WMMA_BP_PREFETCH_A0_WN)
                        next_a0 = a_vectors[2 * tid];
                    if(wn == WMMA_BP_PREFETCH_A1_WN)
                        next_a1 = a_vectors[2 * tid + 1];
                    if(wn == WMMA_BP_PREFETCH_B_WN)
                        next_b = b_vectors[tid];
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
                        wmma<false>(a_frag[wm], b_frag, c_n[wm][wn]);
                    else
                        wmma<true>(a_frag[wm], b_frag, c_n[wm][wn - 2]);
                }
            }
        };

        if constexpr(WMMA_BP_K2_RING)
        {
            constexpr int a_slice_elements = block_m * wmma_tile;
            constexpr int b_slice_elements = block_n * wmma_tile;
            constexpr int total_slices_per_tile = 2;
            const int total_slices = K / wmma_tile;
            for(int slice = 0; slice < total_slices - 2; ++slice)
            {
                const int next_slice = slice + 2;
                const int next_tile = next_slice / total_slices_per_tile;
                const int next_slot = next_slice & 1;
                const half* next_a = a_tile
                    + next_tile * a_tile_elements
                    + next_slot * a_slice_elements;
                const half* next_b_ptr = b_tile
                    + next_tile * b_tile_elements
                    + next_slot * b_slice_elements;
                compute_ring.template operator()<true>(
                    slice & 1, next_a, next_b_ptr);
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
                commit_ring_slice(slice & 1);
            }
            compute_ring.template operator()<false>(
                (total_slices - 2) & 1, nullptr, nullptr);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            __syncthreads();
            compute_ring.template operator()<false>(
                (total_slices - 1) & 1, nullptr, nullptr);
        }
        else if constexpr(WMMA_BP_K_SLICES == 2)
        {
            for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
            {
                const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
                const half* next_b_ptr = b_tile + (k_tile + 1) * b_tile_elements;
                compute_k2.template operator()<true>(next_a, next_b_ptr);
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
                commit_k2();
                __builtin_amdgcn_s_waitcnt(0x7f);
                __syncthreads();
            }
            compute_k2.template operator()<false>(nullptr, nullptr);
        }
        else if constexpr(WMMA_BP_HYBRID_B_PINGPONG)
        {
            int current_b_buffer = 0;
            for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
            {
                const int next_b_buffer = 1 - current_b_buffer;
                const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
                const half* next_b_ptr
                    = b_tile + (k_tile + 1) * b_tile_elements;
                compute_hybrid_b.template operator()<true>(
                    current_b_buffer, next_b_buffer, next_a, next_b_ptr);

                // A alone aliases the operand tile consumed above.  The B
                // refill has already targeted its inactive buffer.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
                commit_a_256();
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
                current_b_buffer = next_b_buffer;
            }
            compute_hybrid_b.template operator()<false>(
                current_b_buffer, 0, nullptr, nullptr);
        }
        else if constexpr(WMMA_BP_HYBRID_A_PINGPONG)
        {
            int current_a_buffer = 0;
            for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
            {
                const int next_a_buffer = 1 - current_a_buffer;
                const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
                const half* next_b_ptr
                    = b_tile + (k_tile + 1) * b_tile_elements;
                compute_hybrid_a.template operator()<true>(
                    current_a_buffer, next_a_buffer, next_a, next_b_ptr);

                // B is the only operand whose destination aliases the tile
                // consumed above.  Retain the first barrier for that hazard;
                // A has already been written to the inactive buffer.
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                __syncthreads();
                commit_b_256();
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
                current_a_buffer = next_a_buffer;
            }
            compute_hybrid_a.template operator()<false>(
                current_a_buffer, 0, nullptr, nullptr);
        }
        else if constexpr(!WMMA_BP_DOUBLE_BUFFER)
        {
            #pragma unroll 1
            for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
            {
                const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
                const half* next_b_ptr = b_tile + (k_tile + 1) * b_tile_elements;
                // CK's gfx11 inter-wave WMMA scheduler raises priority for the
                // MAC cluster, then drops it before the LDS handoff.  Keep the
                // experiment opt-in: all waves still execute identical work,
                // and the surrounding barriers retain the existing ordering.
                if constexpr(WMMA_BP_SET_PRIO)
                {
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_setprio(1);
                    __builtin_amdgcn_sched_barrier(0);
                }
                compute_tile.template operator()<true>(next_a, next_b_ptr);
                if constexpr(WMMA_BP_SET_PRIO)
                {
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_setprio(0);
                    __builtin_amdgcn_sched_barrier(0);
                }

                if constexpr(WMMA_BP_SPLIT_BARRIER)
                {
                    asm volatile("s_barrier_signal -1" ::: "memory");
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    asm volatile("s_barrier_wait -1" ::: "memory");
                }
                else if constexpr(WMMA_BP_WAIT_AFTER_BARRIER)
                {
                    __syncthreads();
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                }
                else if constexpr(WMMA_BP_NO_EXPLICIT_VMWAIT)
                    __syncthreads();
                else
                {
                    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                    __syncthreads();
                }
                if constexpr(WMMA_BP_LATE_B_REFILL)
                {
                    commit_a_128();
                    refill_and_commit_b_128(next_b_ptr);
                }
                else
                    commit_all();
                __builtin_amdgcn_s_waitcnt(0x7f);
                __syncthreads();
            }
            if constexpr(WMMA_BP_SET_PRIO)
            {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(1);
                __builtin_amdgcn_sched_barrier(0);
            }
            compute_tile.template operator()<false>(nullptr, nullptr);
            if constexpr(WMMA_BP_SET_PRIO)
            {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        else if constexpr(WMMA_BP_DOUBLE_BUFFER_LATE)
        {
            static_assert(!WMMA_BP_DOUBLE_BUFFER_LATE
                              || (WMMA_BP_PACK_N && block_m == 256
                                  && block_n == 128),
                          "late-refill double buffer specializes N-packed 256x128");
            auto compute_late = [&]<int buffer>()
            {
                fragment<half, wmma_tile> a_frag[4];
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    const half* source = a_lds + buffer * block_m * stride_a
                        + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                    load_matrix<m_input::matrix_a, m_layout::row_major>(
                        a_frag[wm], source, stride_a, block_k);
                }

                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    fragment<half, wmma_tile> b_frag;
                    const half* source = b_lds + buffer * block_n * stride_b
                        + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                    load_matrix<m_input::matrix_b, m_layout::col_major>(
                        b_frag, source, block_k, stride_b);
                    #pragma unroll
                    for(int wm = 0; wm < 4; ++wm)
                    {
                        if(wn < 2)
                            wmma<false>(a_frag[wm], b_frag, c_n[wm][wn]);
                        else
                            wmma<true>(a_frag[wm], b_frag, c_n[wm][wn - 2]);
                    }
                }
            };

            auto refill_late = [&]<int next_buffer>(const half* next_a,
                                                    const half* next_b_ptr)
            {
                const u16x8* a_vectors = reinterpret_cast<const u16x8*>(next_a);
                const u16x8* b_vectors
                    = reinterpret_cast<const u16x8*>(next_b_ptr);
                next_a0 = a_vectors[2 * tid];
                next_a1 = a_vectors[2 * tid + 1];
                next_b = b_vectors[tid];
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                *reinterpret_cast<u16x8*>(
                    a_lds + next_buffer * block_m * stride_a
                        + tid * stride_a) = next_a0;
                *reinterpret_cast<u16x8*>(
                    a_lds + next_buffer * block_m * stride_a
                        + tid * stride_a + 8) = next_a1;
                const int b_row = tid >> 1;
                const int b_half = (tid & 1) * 8;
                *reinterpret_cast<u16x8*>(
                    b_lds + next_buffer * block_n * stride_b
                        + b_row * stride_b + b_half) = next_b;
            };

            int k_tile = 0;
            for(; k_tile + 2 < k_tiles; k_tile += 2)
            {
                compute_late.template operator()<0>();
                refill_late.template operator()<1>(
                    a_tile + (k_tile + 1) * a_tile_elements,
                    b_tile + (k_tile + 1) * b_tile_elements);
                __builtin_amdgcn_s_waitcnt(0x7f);
                __syncthreads();

                compute_late.template operator()<1>();
                refill_late.template operator()<0>(
                    a_tile + (k_tile + 2) * a_tile_elements,
                    b_tile + (k_tile + 2) * b_tile_elements);
                __builtin_amdgcn_s_waitcnt(0x7f);
                __syncthreads();
            }

            compute_late.template operator()<0>();
            if(k_tile < k_tiles - 1)
            {
                refill_late.template operator()<1>(
                    a_tile + (k_tile + 1) * a_tile_elements,
                    b_tile + (k_tile + 1) * b_tile_elements);
                __builtin_amdgcn_s_waitcnt(0x7f);
                __syncthreads();
                compute_late.template operator()<1>();
            }
        }
        else
        {
            static_assert(!WMMA_BP_DOUBLE_BUFFER
                              || (block_m == 256 && block_n == 128),
                          "double-buffer block prepack currently uses a 256x128 block");
            int current_buffer = 0;

            auto compute_double = [&]<bool do_prefetch>(
                                      const half* next_a,
                                      const half* next_b_ptr,
                                      int current_buffer_for_step,
                                      int next_buffer)
            {
                static_assert(!WMMA_BP_DOUBLE_BUFFER || !WMMA_BP_PACK_N
                                  || WMMA_BP_DOUBLE_BUFFER_LATE,
                              "double-buffer block prepack currently packs C along M");
                fragment<half, wmma_tile> b_frag[4];
                #pragma unroll
                for(int wn = 0; wn < 4; ++wn)
                {
                    const half* source = b_lds
                        + current_buffer_for_step * block_n * stride_b
                        + (warp_n_base + wn * wmma_tile + half_lane) * stride_b;
                    load_matrix<m_input::matrix_b, m_layout::col_major>(
                        b_frag[wn], source, block_k, stride_b);
                }

                u16x8 staged;
                #pragma unroll
                for(int wm = 0; wm < 4; ++wm)
                {
                    if constexpr(do_prefetch)
                    {
                        if(wm == 0)
                        {
                            const u16x8* vectors
                                = reinterpret_cast<const u16x8*>(next_a);
                            staged = vectors[2 * tid];
                        }
                        else if(wm == 1)
                        {
                            const u16x8* vectors
                                = reinterpret_cast<const u16x8*>(next_a);
                            staged = vectors[2 * tid + 1];
                        }
                        else if(wm == 2)
                        {
                            const u16x8* vectors
                                = reinterpret_cast<const u16x8*>(next_b_ptr);
                            staged = vectors[tid];
                        }
                    }

                    fragment<half, wmma_tile> a_frag;
                    const half* source = a_lds
                        + current_buffer_for_step * block_m * stride_a
                        + (warp_m_base + wm * wmma_tile + half_lane) * stride_a;
                    load_matrix<m_input::matrix_a, m_layout::row_major>(
                        a_frag, source, stride_a, block_k);

                    #pragma unroll
                    for(int wn = 0; wn < 4; ++wn)
                    {
                        if(wm < 2)
                            wmma<false>(a_frag, b_frag[wn], c_m[wm][wn]);
                        else
                            wmma<true>(a_frag, b_frag[wn], c_m[wm - 2][wn]);
                    }

                    if constexpr(do_prefetch)
                    {
                        if(wm < 3)
                        {
                            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                            if(wm == 0)
                                *reinterpret_cast<u16x8*>(
                                    a_lds + next_buffer * block_m * stride_a
                                        + tid * stride_a) = staged;
                            else if(wm == 1)
                                *reinterpret_cast<u16x8*>(
                                    a_lds + next_buffer * block_m * stride_a
                                        + tid * stride_a + 8) = staged;
                            else
                            {
                                const int b_row = tid >> 1;
                                const int b_half = (tid & 1) * 8;
                                *reinterpret_cast<u16x8*>(
                                    b_lds + next_buffer * block_n * stride_b
                                        + b_row * stride_b + b_half) = staged;
                            }
                        }
                    }
                }
            };

            if constexpr(WMMA_BP_DOUBLE_BUFFER_STATIC)
            {
                int k_tile = 0;
                for(; k_tile + 2 < k_tiles; k_tile += 2)
                {
                    compute_double.template operator()<true>(
                        a_tile + (k_tile + 1) * a_tile_elements,
                        b_tile + (k_tile + 1) * b_tile_elements,
                        0,
                        1);
                    __builtin_amdgcn_s_waitcnt(0x7f);
                    __syncthreads();

                    compute_double.template operator()<true>(
                        a_tile + (k_tile + 2) * a_tile_elements,
                        b_tile + (k_tile + 2) * b_tile_elements,
                        1,
                        0);
                    __builtin_amdgcn_s_waitcnt(0x7f);
                    __syncthreads();
                }

                if(k_tile < k_tiles - 1)
                {
                    compute_double.template operator()<true>(
                        a_tile + (k_tile + 1) * a_tile_elements,
                        b_tile + (k_tile + 1) * b_tile_elements,
                        0,
                        1);
                    __builtin_amdgcn_s_waitcnt(0x7f);
                    __syncthreads();
                    current_buffer = 1;
                }
                compute_double.template operator()<false>(
                    nullptr, nullptr, current_buffer, 0);
            }
            else
            {
                for(int k_tile = 0; k_tile < k_tiles - 1; ++k_tile)
                {
                    const half* next_a = a_tile + (k_tile + 1) * a_tile_elements;
                    const half* next_b_ptr = b_tile + (k_tile + 1) * b_tile_elements;
                    compute_double.template operator()<true>(
                        next_a, next_b_ptr, current_buffer, 1 - current_buffer);
                    __builtin_amdgcn_s_waitcnt(0x7f);
                    __syncthreads();
                    current_buffer = 1 - current_buffer;
                }
                compute_double.template operator()<false>(
                    nullptr, nullptr, current_buffer, 0);
            }
        }

#if WMMA_BP_VECTOR_EPILOGUE
        auto store_vector_tile = [&]<bool opsel>(
                                     fragment<half, wmma_tile>& frag,
                                     int tile_row,
                                     int tile_col)
        {
            static_assert(WMMA_BP_PACK_N,
                          "vector epilogue expects N-packed accumulators");
            uint32_t value[8];
            #pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                value[i] = static_cast<uint32_t>(__half_as_ushort(
                    frag.get()[2 * i + (opsel ? 1 : 0)]));
            }

            auto dpp_xor = []<int mask, int bank_mask>(uint32_t keep,
                                                        uint32_t peer)
            {
                // RDNA 3.5 bank_mask selects four-lane groups, not individual
                // lane-id bits.  Bits 0/1 therefore require a full-bank DPP
                // followed by lane selection; bit 2 maps to bank groups and
                // can select directly with 0xa/0x5.
                uint32_t result = keep;
                if constexpr(mask == 1 && bank_mask == 0xf)
                    asm volatile(
                        "v_mov_b32_dpp %0, %1 row_xmask:1 row_mask:0xf bank_mask:0xf bound_ctrl:0"
                        : "+v"(result)
                        : "v"(peer));
                else if constexpr(mask == 2 && bank_mask == 0xf)
                    asm volatile(
                        "v_mov_b32_dpp %0, %1 row_xmask:2 row_mask:0xf bank_mask:0xf bound_ctrl:0"
                        : "+v"(result)
                        : "v"(peer));
                else if constexpr(mask == 4 && bank_mask == 0xa)
                    asm volatile(
                        "v_mov_b32_dpp %0, %1 row_xmask:4 row_mask:0xf bank_mask:0xa bound_ctrl:0"
                        : "+v"(result)
                        : "v"(peer));
                else
                {
                    static_assert(mask == 4 && bank_mask == 0x5);
                    asm volatile(
                        "v_mov_b32_dpp %0, %1 row_xmask:4 row_mask:0xf bank_mask:0x5 bound_ctrl:0"
                        : "+v"(result)
                        : "v"(peer));
                }
                return result;
            };

            const bool lane_bit_0 = (lane & 1) != 0;
            #pragma unroll
            for(int pair = 0; pair < 4; ++pair)
            {
                const int low = 2 * pair;
                const uint32_t a = value[low];
                const uint32_t b = value[low + 1];
                const uint32_t peer_b
                    = dpp_xor.template operator()<1, 0xf>(b, b);
                const uint32_t peer_a
                    = dpp_xor.template operator()<1, 0xf>(a, a);
                value[low] = lane_bit_0 ? peer_b : a;
                value[low + 1] = lane_bit_0 ? b : peer_a;
            }
            const bool lane_bit_1 = (lane & 2) != 0;
            #pragma unroll
            for(int group = 0; group < 2; ++group)
            {
                #pragma unroll
                for(int inner = 0; inner < 2; ++inner)
                {
                    const int low = group * 4 + inner;
                    const uint32_t a = value[low];
                    const uint32_t b = value[low + 2];
                    const uint32_t peer_b
                        = dpp_xor.template operator()<2, 0xf>(b, b);
                    const uint32_t peer_a
                        = dpp_xor.template operator()<2, 0xf>(a, a);
                    value[low] = lane_bit_1 ? peer_b : a;
                    value[low + 2] = lane_bit_1 ? b : peer_a;
                }
            }

            #pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                const uint32_t low = value[i];
                const uint32_t high = value[i + 4];
                value[i]
                    = dpp_xor.template operator()<4, 0xa>(low, high);
                value[i + 4]
                    = dpp_xor.template operator()<4, 0x5>(high, low);
            }

            u16x8 packed;
            #pragma unroll
            for(int i = 0; i < 8; ++i)
                packed[i] = static_cast<uint16_t>(value[i]);

            const int output_row
                = tile_row + 2 * (lane & 7) + (lane >> 4);
            const int output_col = tile_col + 8 * ((lane >> 3) & 1);
            *reinterpret_cast<u16x8*>(
                C + static_cast<size_t>(output_row) * N + output_col)
                = packed;
        };
#endif

#if WMMA_BP_FULL_TILE_STORE
        // The fixed-shape benchmark is exactly divisible by the packed tile.
        // Keep this specialization source-defined rather than deleting the
        // generated exec masks: callers must pass the full 4096x4096 shape.
        static_assert(block_m == 256 && block_n == 128 && WMMA_BP_PACK_N,
                      "full-tile store is specific to the record geometry");
        auto store_full_tile = [&]<bool opsel>(fragment<half, wmma_tile>& frag,
                                               int row,
                                               int col)
        {
            #pragma unroll
            for(int i = 0; i < wmma_tile / 2; ++i)
                C[static_cast<size_t>(row + 2 * i) * static_cast<size_t>(N)
                  + static_cast<size_t>(col)]
                    = frag[2 * i + (opsel ? 1 : 0)];
        };
#endif

        #pragma unroll
        for(int wm = 0; wm < warp_tile_m; ++wm)
        {
            #pragma unroll
            for(int wn = 0; wn < 4; ++wn)
            {
#if WMMA_BP_VECTOR_EPILOGUE
                if(wn < 2)
                    store_vector_tile.template operator()<false>(
                        c_n[wm][wn],
                        block_row + warp_m_base + wm * wmma_tile,
                        block_col + warp_n_base + wn * wmma_tile);
                else
                    store_vector_tile.template operator()<true>(
                        c_n[wm][wn - 2],
                        block_row + warp_m_base + wm * wmma_tile,
                        block_col + warp_n_base + wn * wmma_tile);
#else
                if constexpr(!WMMA_BP_PACK_N)
                {
                    if(wm < 2)
                        store_matrix<m_layout::row_major, false, false>(
                            C,
                            c_m[wm][wn],
                            block_row + warp_m_base + wm * wmma_tile + half_wave,
                            block_col + warp_n_base + wn * wmma_tile + half_lane,
                            M,
                            N);
                    else
                        store_matrix<m_layout::row_major, false, true>(
                            C,
                            c_m[wm - 2][wn],
                            block_row + warp_m_base + wm * wmma_tile + half_wave,
                            block_col + warp_n_base + wn * wmma_tile + half_lane,
                            M,
                            N);
                }
                else
                {
                    if(wn < 2)
#if WMMA_BP_FULL_TILE_STORE
                        store_full_tile.template operator()<false>(
                            c_n[wm][wn],
                            block_row + warp_m_base + wm * wmma_tile + half_wave,
                            block_col + warp_n_base + wn * wmma_tile + half_lane);
#else
                        store_matrix<m_layout::row_major, false, false>(
                            C,
                            c_n[wm][wn],
                            block_row + warp_m_base + wm * wmma_tile + half_wave,
                            block_col + warp_n_base + wn * wmma_tile + half_lane,
                            M,
                            N);
#endif
                    else
#if WMMA_BP_FULL_TILE_STORE
                        store_full_tile.template operator()<true>(
                            c_n[wm][wn - 2],
                            block_row + warp_m_base + wm * wmma_tile + half_wave,
                            block_col + warp_n_base + wn * wmma_tile + half_lane);
#else
                        store_matrix<m_layout::row_major, false, true>(
                            C,
                            c_n[wm][wn - 2],
                            block_row + warp_m_base + wm * wmma_tile + half_wave,
                            block_col + warp_n_base + wn * wmma_tile + half_lane,
                            M,
                            N);
#endif
                }
#endif
            }
        }
    }
};

} // namespace rocm_wmma_gemm

#endif
