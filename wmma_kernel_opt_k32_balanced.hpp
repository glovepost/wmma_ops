// ============================================================================
// Load-balanced variant of the 128x128x32 gfx1151 leader.
//
// The ordinary K32 kernel assigns 128 threads to vector-staged A and 128 to
// scalar-transposed B. B takes far more LDS instructions, leaving A-loader
// waves waiting at the work-group barrier. This variant assigns two waves to
// A and six to B while preserving the same output geometry and arithmetic.
// ============================================================================

#ifndef WMMA_KERNEL_OPT_K32_BALANCED_HPP
#define WMMA_KERNEL_OPT_K32_BALANCED_HPP

#include "wmma_kernel_opt_k32.hpp"

template<typename cfg>
__device__ __forceinline__ void balanced_stage_b_vector(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int k_local,
    int n_base,
    half8_k32 value
) {
    union {
        half8_k32 packed;
        _Float16 element[8];
    } unpacked;
    unpacked.packed = value;
    #pragma unroll
    for (int n_local = 0; n_local < 8; ++n_local) {
        b_lds[n_base + n_local][k_local] =
            *reinterpret_cast<__half*>(&unpacked.element[n_local]);
    }
}

__launch_bounds__(OptK32Config::NUM_THREADS)
__global__ void wmma_gemm_kernel_opt_k32_balanced(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    int n,
    int k_size
) {
    using cfg = OptK32Config;
    constexpr int A_THREADS = 64;
    constexpr int B_THREADS = cfg::NUM_THREADS - A_THREADS;

    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const bool loads_a = tid < A_THREADS;
    const int load_id = loads_a ? tid : tid - A_THREADS;
    const int warp_m = warp / cfg::WARPS_N;
    const int warp_n = warp % cfg::WARPS_N;
    const int warp_m_base = warp_m * cfg::WARP_TILE_M * cfg::WMMA_M;
    const int warp_n_base = warp_n * cfg::WARP_TILE_N * cfg::WMMA_N;
    const int block_m = blockIdx.y * cfg::BLOCK_M;
    const int block_n = blockIdx.x * cfg::BLOCK_N;

    fragment<accumulator, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, float>
        accum[cfg::WARP_TILE_M][cfg::WARP_TILE_N];
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            fill_fragment(accum[tile_m][tile_n], 0.0f);
        }
    }

    if (loads_a) {
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int vector_index = load_id + slot * A_THREADS;
            const int row = vector_index >> 2;
            const int k_vector = vector_index & 3;
            const half8_k32 value =
                *reinterpret_cast<const half8_k32*>(
                    a + (block_m + row) * k_size + k_vector * 8);
            *reinterpret_cast<half8_k32*>(
                &a_lds[0][row][k_vector * 8]) = value;
        }
    } else {
        #pragma unroll
        for (int slot = 0; slot < 3; ++slot) {
            const int vector_index = load_id + slot * B_THREADS;
            if (vector_index < 512) {
                const int k_local = vector_index >> 4;
                const int n_base = (vector_index & 15) << 3;
                const half8_k32 value =
                    *reinterpret_cast<const half8_k32*>(
                        b + k_local * n + block_n + n_base);
                balanced_stage_b_vector<cfg>(
                    b_lds[0], k_local, n_base, value);
            }
        }
    }
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        const int next = 1 - current;
        const bool has_next = k + cfg::BLOCK_K < k_size;
        half8_k32 prefetched[8] = {};
        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, row_major> a_frag[cfg::WARP_TILE_M];
        fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, col_major> b_frag[cfg::WARP_TILE_N];

        k32_load_fragments<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, 0);

        int mma_index = 0;
        #pragma unroll
        for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
            #pragma unroll
            for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
                mma_sync(accum[tile_m][tile_n], a_frag[tile_m],
                         b_frag[tile_n], accum[tile_m][tile_n]);
                if (has_next) {
                    if (loads_a) {
                        const int vector_index =
                            load_id + mma_index * A_THREADS;
                        const int row = vector_index >> 2;
                        const int k_vector = vector_index & 3;
                        prefetched[mma_index] =
                            *reinterpret_cast<const half8_k32*>(
                                a + (block_m + row) * k_size + k
                                    + cfg::BLOCK_K + k_vector * 8);
                    } else if (mma_index < 3) {
                        const int vector_index =
                            load_id + mma_index * B_THREADS;
                        if (vector_index < 512) {
                            const int k_local = vector_index >> 4;
                            const int n_base = (vector_index & 15) << 3;
                            prefetched[mma_index] =
                                *reinterpret_cast<const half8_k32*>(
                                    b + (k + cfg::BLOCK_K + k_local) * n
                                        + block_n + n_base);
                        }
                    }
                }
                ++mma_index;
            }
        }

        k32_load_fragments<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, cfg::WMMA_K);
        k32_mma_slice<cfg>(accum, a_frag, b_frag);

        if (has_next) {
            if (loads_a) {
                #pragma unroll
                for (int slot = 0; slot < 8; ++slot) {
                    const int vector_index = load_id + slot * A_THREADS;
                    const int row = vector_index >> 2;
                    const int k_vector = vector_index & 3;
                    *reinterpret_cast<half8_k32*>(
                        &a_lds[next][row][k_vector * 8]) = prefetched[slot];
                }
            } else {
                #pragma unroll
                for (int slot = 0; slot < 3; ++slot) {
                    const int vector_index = load_id + slot * B_THREADS;
                    if (vector_index < 512) {
                        const int k_local = vector_index >> 4;
                        const int n_base = (vector_index & 15) << 3;
                        balanced_stage_b_vector<cfg>(
                            b_lds[next], k_local, n_base, prefetched[slot]);
                    }
                }
            }
        }
        __syncthreads();
        current = next;
    }

    const int column_lane = lane & 15;
    const int row_parity = lane >> 4;
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            const int tile_row =
                block_m + warp_m_base + tile_m * cfg::WMMA_M;
            const int column = block_n + warp_n_base
                + tile_n * cfg::WMMA_N + column_lane;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = tile_row + (i << 1) + row_parity;
                c[row * n + column] = accum[tile_m][tile_n].x[i];
            }
        }
    }
}

#endif
