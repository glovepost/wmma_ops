// ============================================================================
// K=32 interleaved record-shape specialization for gfx1151.
//
// Two independent K=16 WMMA slices share one global-to-LDS stage and one
// work-group barrier.  Four next-tile half8 loads are issued during the first
// slice, leaving twelve WMMA instructions before the prefetched values are
// written to LDS.
// ============================================================================

#ifndef WMMA_KERNEL_OPT_K32_HPP
#define WMMA_KERNEL_OPT_K32_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "rocwmma_patch/rocwmma_gfx1151.hpp"

using namespace rocwmma;

typedef _Float16 half8_k32 __attribute__((ext_vector_type(8)));

struct OptK32Config {
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
    static constexpr int NWARPS = WARPS_M * WARPS_N;
    static constexpr int WARP_TILE_M = 2;
    static constexpr int WARP_TILE_N = 4;
    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 128;
    static constexpr int BLOCK_K = 32;
    static constexpr int LDS_STRIDE_A = BLOCK_K + 8;
    static constexpr int LDS_STRIDE_B = BLOCK_K + 8;
    static constexpr int NUM_THREADS = NWARPS * WARP_SIZE;
    static constexpr int HALF_BLOCK = NUM_THREADS / 2;
};

struct OptK32StreamConfig {
    static constexpr int WMMA_M = OptK32Config::WMMA_M;
    static constexpr int WMMA_N = OptK32Config::WMMA_N;
    static constexpr int WMMA_K = OptK32Config::WMMA_K;
    static constexpr int WARP_SIZE = OptK32Config::WARP_SIZE;
    static constexpr int WARPS_M = OptK32Config::WARPS_M;
    static constexpr int WARPS_N = OptK32Config::WARPS_N;
    static constexpr int NWARPS = OptK32Config::NWARPS;
    static constexpr int WARP_TILE_M = OptK32Config::WARP_TILE_M;
    static constexpr int WARP_TILE_N = OptK32Config::WARP_TILE_N;
    static constexpr int BLOCK_M = OptK32Config::BLOCK_M;
    static constexpr int BLOCK_N = OptK32Config::BLOCK_N;
    static constexpr int BLOCK_K = OptK32Config::BLOCK_K;
    // Exactly 32 KiB for both double-buffered operands.  This permits two
    // resident workgroups where the padded 40 KiB form permits only one; the
    // measured bank-conflict/occupancy tradeoff decides whether it is useful.
    static constexpr int LDS_STRIDE_A = BLOCK_K;
    static constexpr int LDS_STRIDE_B = BLOCK_K;
    static constexpr int NUM_THREADS = OptK32Config::NUM_THREADS;
    static constexpr int HALF_BLOCK = OptK32Config::HALF_BLOCK;
};

template<typename cfg>
__device__ __forceinline__ void k32_stage_a(
    __half a_lds[][cfg::LDS_STRIDE_A],
    int row,
    const half8_k32 (&value)[4]
) {
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        *reinterpret_cast<half8_k32*>(&a_lds[row][vector * 8]) = value[vector];
    }
}

template<typename cfg>
__device__ __forceinline__ void k32_stage_b(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int lane8,
    int n_base,
    const half8_k32 (&value)[4]
) {
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        union {
            half8_k32 packed;
            _Float16 element[8];
        } unpacked;
        unpacked.packed = value[vector];
        #pragma unroll
        for (int n_local = 0; n_local < 8; ++n_local) {
            b_lds[n_base + n_local][vector * 8 + lane8] =
                *reinterpret_cast<__half*>(&unpacked.element[n_local]);
        }
    }
}

template<typename cfg>
__device__ __forceinline__ void k32_stage_b_prepacked(
    __half b_lds[][cfg::LDS_STRIDE_B],
    int row,
    const half8_k32 (&value)[4]
) {
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        *reinterpret_cast<half8_k32*>(&b_lds[row][vector * 8]) = value[vector];
    }
}

template<typename cfg>
__device__ __forceinline__ void k32_load_fragments(
    fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
             __half, row_major> (&a_frag)[cfg::WARP_TILE_M],
    fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
             __half, col_major> (&b_frag)[cfg::WARP_TILE_N],
    const __half a_lds[][cfg::LDS_STRIDE_A],
    const __half b_lds[][cfg::LDS_STRIDE_B],
    int warp_m_base,
    int warp_n_base,
    int k_subtile
) {
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        load_matrix_sync_lds(
            a_frag[tile_m],
            &a_lds[warp_m_base + tile_m * cfg::WMMA_M][k_subtile],
            cfg::LDS_STRIDE_A);
    }
    #pragma unroll
    for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
        load_matrix_sync_lds_b_transposed(
            b_frag[tile_n],
            &b_lds[warp_n_base + tile_n * cfg::WMMA_N][k_subtile],
            cfg::LDS_STRIDE_B);
    }
}

template<typename cfg>
__device__ __forceinline__ void k32_mma_slice(
    fragment<accumulator, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K, float>
        (&accum)[cfg::WARP_TILE_M][cfg::WARP_TILE_N],
    const fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                   __half, row_major> (&a_frag)[cfg::WARP_TILE_M],
    const fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                   __half, col_major> (&b_frag)[cfg::WARP_TILE_N]
) {
    #pragma unroll
    for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
        #pragma unroll
        for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
            mma_sync(accum[tile_m][tile_n], a_frag[tile_m],
                     b_frag[tile_n], accum[tile_m][tile_n]);
        }
    }
}

template<bool PrepackedB = false>
__launch_bounds__(OptK32Config::NUM_THREADS)
__global__ void wmma_gemm_kernel_opt_k32(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    const int n,
    const int k_size
) {
    using cfg = OptK32Config;

    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const int cid = tid & (cfg::HALF_BLOCK - 1);
    const bool loads_a = tid < cfg::HALF_BLOCK;
    const int lane8 = cid & 7;
    const int n_base = (cid >> 3) << 3;
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

    half8_k32 prologue[4];
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        if (loads_a) {
            prologue[vector] = *reinterpret_cast<const half8_k32*>(
                a + (block_m + cid) * k_size + vector * 8);
        } else {
            if constexpr (PrepackedB) {
                prologue[vector] = *reinterpret_cast<const half8_k32*>(
                    b + (block_n + cid) * k_size + vector * 8);
            } else {
                prologue[vector] = *reinterpret_cast<const half8_k32*>(
                    b + (vector * 8 + lane8) * n + block_n + n_base);
            }
        }
    }
    if (loads_a) {
        k32_stage_a<cfg>(a_lds[0], cid, prologue);
    } else {
        if constexpr (PrepackedB) {
            k32_stage_b_prepacked<cfg>(b_lds[0], cid, prologue);
        } else {
            k32_stage_b<cfg>(b_lds[0], lane8, n_base, prologue);
        }
    }
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        const int next = 1 - current;
        const bool has_next = k + cfg::BLOCK_K < k_size;
        half8_k32 prefetched[4] = {};
        fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, row_major> a_frag[cfg::WARP_TILE_M];
        fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                 __half, col_major> b_frag[cfg::WARP_TILE_N];

        k32_load_fragments<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, 0);

        // First K slice, with the four next-panel loads spread across its
        // first four independent accumulator chains.
        mma_sync(accum[0][0], a_frag[0], b_frag[0], accum[0][0]);
        if (has_next) {
            if (loads_a) {
                prefetched[0] = *reinterpret_cast<const half8_k32*>(
                    a + (block_m + cid) * k_size + k + cfg::BLOCK_K);
            } else if constexpr (PrepackedB) {
                prefetched[0] = *reinterpret_cast<const half8_k32*>(
                    b + (block_n + cid) * k_size + k + cfg::BLOCK_K);
            } else {
                prefetched[0] = *reinterpret_cast<const half8_k32*>(
                    b + (k + cfg::BLOCK_K + lane8) * n + block_n + n_base);
            }
        }
        mma_sync(accum[0][1], a_frag[0], b_frag[1], accum[0][1]);
        if (has_next) {
            if (loads_a) {
                prefetched[1] = *reinterpret_cast<const half8_k32*>(
                    a + (block_m + cid) * k_size + k + cfg::BLOCK_K + 8);
            } else if constexpr (PrepackedB) {
                prefetched[1] = *reinterpret_cast<const half8_k32*>(
                    b + (block_n + cid) * k_size + k + cfg::BLOCK_K + 8);
            } else {
                prefetched[1] = *reinterpret_cast<const half8_k32*>(
                    b + (k + cfg::BLOCK_K + 8 + lane8) * n + block_n + n_base);
            }
        }
        mma_sync(accum[0][2], a_frag[0], b_frag[2], accum[0][2]);
        if (has_next) {
            if (loads_a) {
                prefetched[2] = *reinterpret_cast<const half8_k32*>(
                    a + (block_m + cid) * k_size + k + cfg::BLOCK_K + 16);
            } else if constexpr (PrepackedB) {
                prefetched[2] = *reinterpret_cast<const half8_k32*>(
                    b + (block_n + cid) * k_size + k + cfg::BLOCK_K + 16);
            } else {
                prefetched[2] = *reinterpret_cast<const half8_k32*>(
                    b + (k + cfg::BLOCK_K + 16 + lane8) * n + block_n + n_base);
            }
        }
        mma_sync(accum[0][3], a_frag[0], b_frag[3], accum[0][3]);
        if (has_next) {
            if (loads_a) {
                prefetched[3] = *reinterpret_cast<const half8_k32*>(
                    a + (block_m + cid) * k_size + k + cfg::BLOCK_K + 24);
            } else if constexpr (PrepackedB) {
                prefetched[3] = *reinterpret_cast<const half8_k32*>(
                    b + (block_n + cid) * k_size + k + cfg::BLOCK_K + 24);
            } else {
                prefetched[3] = *reinterpret_cast<const half8_k32*>(
                    b + (k + cfg::BLOCK_K + 24 + lane8) * n + block_n + n_base);
            }
        }
        mma_sync(accum[1][0], a_frag[1], b_frag[0], accum[1][0]);
        mma_sync(accum[1][1], a_frag[1], b_frag[1], accum[1][1]);
        mma_sync(accum[1][2], a_frag[1], b_frag[2], accum[1][2]);
        mma_sync(accum[1][3], a_frag[1], b_frag[3], accum[1][3]);

        k32_load_fragments<cfg>(
            a_frag, b_frag, a_lds[current], b_lds[current],
            warp_m_base, warp_n_base, cfg::WMMA_K);
        k32_mma_slice<cfg>(accum, a_frag, b_frag);

        if (has_next) {
            if (loads_a) {
                k32_stage_a<cfg>(a_lds[next], cid, prefetched);
            } else {
                if constexpr (PrepackedB) {
                    k32_stage_b_prepacked<cfg>(
                        b_lds[next], cid, prefetched);
                } else {
                    k32_stage_b<cfg>(
                        b_lds[next], lane8, n_base, prefetched);
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

// Stream one B fragment at a time through both A rows.  This retains the same
// LDS reads and per-output K accumulation order as the kernel above, but lets
// the compiler recycle the B fragment registers instead of keeping four B
// fragments live across each K slice.
__launch_bounds__(OptK32StreamConfig::NUM_THREADS)
__global__ void wmma_gemm_kernel_opt_k32_streamed(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ c,
    const int n,
    const int k_size
) {
    using cfg = OptK32StreamConfig;

    __shared__ __half a_lds[2][cfg::BLOCK_M][cfg::LDS_STRIDE_A];
    __shared__ __half b_lds[2][cfg::BLOCK_N][cfg::LDS_STRIDE_B];

    const int tid = threadIdx.x;
    const int warp = tid / cfg::WARP_SIZE;
    const int lane = tid & (cfg::WARP_SIZE - 1);
    const int cid = tid & (cfg::HALF_BLOCK - 1);
    const bool loads_a = tid < cfg::HALF_BLOCK;
    const int lane8 = cid & 7;
    const int n_base = (cid >> 3) << 3;
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

    half8_k32 prologue[4];
    #pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
        if (loads_a) {
            prologue[vector] = *reinterpret_cast<const half8_k32*>(
                a + (block_m + cid) * k_size + vector * 8);
        } else {
            prologue[vector] = *reinterpret_cast<const half8_k32*>(
                b + (vector * 8 + lane8) * n + block_n + n_base);
        }
    }
    if (loads_a) {
        k32_stage_a<cfg>(a_lds[0], cid, prologue);
    } else {
        k32_stage_b<cfg>(b_lds[0], lane8, n_base, prologue);
    }
    __syncthreads();

    int current = 0;
    #pragma unroll 1
    for (int k = 0; k < k_size; k += cfg::BLOCK_K) {
        const int next = 1 - current;
        const bool has_next = k + cfg::BLOCK_K < k_size;
        half8_k32 prefetched[4] = {};

        #pragma unroll
        for (int k_subtile = 0; k_subtile < cfg::BLOCK_K;
             k_subtile += cfg::WMMA_K) {
            fragment<matrix_a, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                     __half, row_major> a_frag[cfg::WARP_TILE_M];
            #pragma unroll
            for (int tile_m = 0; tile_m < cfg::WARP_TILE_M; ++tile_m) {
                load_matrix_sync_lds(
                    a_frag[tile_m],
                    &a_lds[current]
                          [warp_m_base + tile_m * cfg::WMMA_M][k_subtile],
                    cfg::LDS_STRIDE_A);
            }

            #pragma unroll
            for (int tile_n = 0; tile_n < cfg::WARP_TILE_N; ++tile_n) {
                fragment<matrix_b, cfg::WMMA_M, cfg::WMMA_N, cfg::WMMA_K,
                         __half, col_major> b_frag;
                load_matrix_sync_lds_b_transposed(
                    b_frag,
                    &b_lds[current]
                          [warp_n_base + tile_n * cfg::WMMA_N][k_subtile],
                    cfg::LDS_STRIDE_B);

                mma_sync(accum[0][tile_n], a_frag[0], b_frag,
                         accum[0][tile_n]);
                if (has_next && k_subtile == 0 && tile_n < 2) {
                    const int vector = tile_n * 2;
                    if (loads_a) {
                        prefetched[vector] =
                            *reinterpret_cast<const half8_k32*>(
                                a + (block_m + cid) * k_size + k
                                    + cfg::BLOCK_K + vector * 8);
                    } else {
                        prefetched[vector] =
                            *reinterpret_cast<const half8_k32*>(
                                b + (k + cfg::BLOCK_K + vector * 8 + lane8)
                                    * n + block_n + n_base);
                    }
                }

                mma_sync(accum[1][tile_n], a_frag[1], b_frag,
                         accum[1][tile_n]);
                if (has_next && k_subtile == 0 && tile_n < 2) {
                    const int vector = tile_n * 2 + 1;
                    if (loads_a) {
                        prefetched[vector] =
                            *reinterpret_cast<const half8_k32*>(
                                a + (block_m + cid) * k_size + k
                                    + cfg::BLOCK_K + vector * 8);
                    } else {
                        prefetched[vector] =
                            *reinterpret_cast<const half8_k32*>(
                                b + (k + cfg::BLOCK_K + vector * 8 + lane8)
                                    * n + block_n + n_base);
                    }
                }
            }
        }

        if (has_next) {
            if (loads_a) {
                k32_stage_a<cfg>(a_lds[next], cid, prefetched);
            } else {
                k32_stage_b<cfg>(b_lds[next], lane8, n_base, prefetched);
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
