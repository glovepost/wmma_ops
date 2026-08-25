// Reproducible standalone harness for the gfx1151 FP16-output candidate.
//
// Kernel implementation: adelj88/rocm_wmma_gemm at
// 281b5dfd7fbff9cea80753bc55274a54f4a7c53a (MIT).
// This file supplies the exact tuned configuration, timing protocol, and an
// independent full-output rocBLAS reference. It is an isolated optimization
// harness: FP16 output is a different numerical contract from wmma-ops' FP32
// record and must not be promoted as the same benchmark class.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocm_wmma_gemm/kernel/kernel.hpp>
#if RECORD_PRODUCER_CONSUMER
#include <rocm_wmma_gemm/kernel/producer_consumer.hpp>
#endif
#if RECORD_DIRECT_PREPACKED
#include <rocm_wmma_gemm/kernel/prepacked_direct.hpp>
#endif
#if RECORD_BLOCK_PREPACKED
#include <rocm_wmma_gemm/kernel/block_prepacked.hpp>
#endif
#if RECORD_BLOCK_K2_RING
#include <rocm_wmma_gemm/kernel/block_k2_ring.hpp>
#endif
#if RECORD_WAVEPRIVATE_A
#include <rocm_wmma_gemm/kernel/waveprivate_a.hpp>
#endif
#if RECORD_DIRECT_A_SHARED_B
#include <rocm_wmma_gemm/kernel/direct_a_shared_b.hpp>
#endif
#if RECORD_PERSISTENT_BLOCK
#include <rocm_wmma_gemm/kernel/persistent_block.hpp>
#endif
#if RECORD_INTERLEAVED_PINGPONG
#include <rocm_wmma_gemm/kernel/interleaved_pingpong.hpp>
#endif
#if RECORD_BLOCK_512X128
#include <rocm_wmma_gemm/kernel/block_512x128.hpp>
#endif
#if RECORD_B_RESIDENT
#include <rocm_wmma_gemm/kernel/b_resident.hpp>
#endif
#if RECORD_DIRECT_B_SHARED_A
#include <rocm_wmma_gemm/kernel/direct_b_shared_a.hpp>
#endif
#if RECORD_PAIRED_A_HANDOFF
#include <rocm_wmma_gemm/kernel/paired_a_handoff.hpp>
#endif
#if RECORD_EMBEDDED_AB_PINGPONG
#include <rocm_wmma_gemm/kernel/embedded_ab_pingpong.hpp>
#endif
#if RECORD_BLOCK_K32_PUBLICATION
#include <rocm_wmma_gemm/kernel/block_k32_publication.hpp>
#endif
#if RECORD_COMPACT_XOR_PINGPONG
#include <rocm_wmma_gemm/kernel/compact_xor_pingpong.hpp>
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        const hipError_t status = (expr);                                      \
        if (status != hipSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                  \
                         hipGetErrorString(status));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define ROCBLAS_CHECK(expr)                                                    \
    do {                                                                       \
        const rocblas_status status = (expr);                                  \
        if (status != rocblas_status_success) {                                \
            std::fprintf(stderr, "%s failed: rocBLAS status %d\n", #expr,   \
                         static_cast<int>(status));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

#ifndef RECORD_WARPS_M
#define RECORD_WARPS_M 2
#endif
#ifndef RECORD_WARPS_N
#define RECORD_WARPS_N 2
#endif
#ifndef RECORD_WARP_TILE_M
#define RECORD_WARP_TILE_M 4
#endif
#ifndef RECORD_WARP_TILE_N
#define RECORD_WARP_TILE_N 4
#endif
#ifndef RECORD_DYNAMIC_LDS_BYTES
#define RECORD_DYNAMIC_LDS_BYTES 0
#endif
#ifndef RECORD_SWIZZLE
#define RECORD_SWIZZLE 16
#endif
#ifndef RECORD_SINGLE_BUFFER
#define RECORD_SINGLE_BUFFER 0
#endif
#ifndef RECORD_BITS
#define RECORD_BITS 128
#endif
#ifndef RECORD_K_SLICES
#define RECORD_K_SLICES 1
#endif
#ifndef RECORD_PREPACKED
#define RECORD_PREPACKED 0
#endif
#ifndef RECORD_PREPACK_A
#define RECORD_PREPACK_A RECORD_PREPACKED
#endif
#ifndef RECORD_PREPACK_B
#define RECORD_PREPACK_B RECORD_PREPACKED
#endif
#ifndef RECORD_PRODUCER_CONSUMER
#define RECORD_PRODUCER_CONSUMER 0
#endif
#ifndef RECORD_DIRECT_PREPACKED
#define RECORD_DIRECT_PREPACKED 0
#endif
#ifndef RECORD_BLOCK_PREPACKED
#define RECORD_BLOCK_PREPACKED 0
#endif
#ifndef RECORD_BLOCK_K2_RING
#define RECORD_BLOCK_K2_RING 0
#endif
#ifndef RECORD_WAVEPRIVATE_A
#define RECORD_WAVEPRIVATE_A 0
#endif
#ifndef RECORD_DIRECT_A_SHARED_B
#define RECORD_DIRECT_A_SHARED_B 0
#endif
#ifndef RECORD_PERSISTENT_BLOCK
#define RECORD_PERSISTENT_BLOCK 0
#endif
#ifndef RECORD_GRID_BLOCKS
#define RECORD_GRID_BLOCKS 0
#endif
#ifndef RECORD_GRID_BLOCKS_B
#define RECORD_GRID_BLOCKS_B RECORD_GRID_BLOCKS
#endif
#ifndef RECORD_INTERLEAVED_PINGPONG
#define RECORD_INTERLEAVED_PINGPONG 0
#endif
#ifndef RECORD_BLOCK_512X128
#define RECORD_BLOCK_512X128 0
#endif
#ifndef RECORD_B_RESIDENT
#define RECORD_B_RESIDENT 0
#endif
#ifndef RECORD_DIRECT_B_SHARED_A
#define RECORD_DIRECT_B_SHARED_A 0
#endif
#ifndef RECORD_PAIRED_A_HANDOFF
#define RECORD_PAIRED_A_HANDOFF 0
#endif
#ifndef RECORD_EMBEDDED_AB_PINGPONG
#define RECORD_EMBEDDED_AB_PINGPONG 0
#endif
#ifndef RECORD_BLOCK_K32_PUBLICATION
#define RECORD_BLOCK_K32_PUBLICATION 0
#endif
#ifndef RECORD_COMPACT_XOR_PINGPONG
#define RECORD_COMPACT_XOR_PINGPONG 0
#endif
#ifndef RECORD_BLOCK_SLICE_MAJOR
#define RECORD_BLOCK_SLICE_MAJOR 0
#endif
#ifndef RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER
#define RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER 1
#endif
#ifndef RECORD_PRODUCER_WAVES
#define RECORD_PRODUCER_WAVES 5
#endif
#ifndef RECORD_M
#define RECORD_M 4096
#endif
#ifndef RECORD_N
#define RECORD_N 4096
#endif
#ifndef RECORD_K
#define RECORD_K 4096
#endif
#ifndef RECORD_PADDED_BLOCK_CONTRACT
#define RECORD_PADDED_BLOCK_CONTRACT 0
#endif
#ifndef RECORD_SPLIT_MODULE_DISPATCH
#define RECORD_SPLIT_MODULE_DISPATCH 0
#endif
#ifndef RECORD_SPLIT_CONCURRENT
#define RECORD_SPLIT_CONCURRENT 0
#endif
#ifndef RECORD_MODULE0_PATH
#define RECORD_MODULE0_PATH ""
#endif
#ifndef RECORD_MODULE1_PATH
#define RECORD_MODULE1_PATH ""
#endif
#define RECORD_BLOCK_CONTRACT \
    (RECORD_BLOCK_PREPACKED | RECORD_BLOCK_K2_RING | RECORD_WAVEPRIVATE_A \
     | RECORD_DIRECT_A_SHARED_B | RECORD_PERSISTENT_BLOCK \
     | RECORD_INTERLEAVED_PINGPONG | RECORD_BLOCK_512X128 \
     | RECORD_B_RESIDENT | RECORD_DIRECT_B_SHARED_A \
     | RECORD_PAIRED_A_HANDOFF | RECORD_EMBEDDED_AB_PINGPONG \
     | RECORD_BLOCK_K32_PUBLICATION | RECORD_COMPACT_XOR_PINGPONG)

constexpr int kM = RECORD_M;
constexpr int kN = RECORD_N;
constexpr int kK = RECORD_K;
constexpr int kWarmupDefault = 200;
constexpr int kIterationsDefault = 100;
constexpr int kTimingBlocks = 5;
constexpr double kPromotionTargetTflops = 50.0;

// Tuned parameter order:
// warps_m, warps_n, warp_tile_m, warp_tile_n, k_slices, single_buffer,
// swizzle, vector bits, alignment fast path.
#if RECORD_COMPACT_XOR_PINGPONG
using RecordKernel = rocm_wmma_gemm::compact_xor_pingpong_gemm;
#elif RECORD_BLOCK_K32_PUBLICATION
using RecordKernel = rocm_wmma_gemm::block_k32_publication_gemm;
#elif RECORD_EMBEDDED_AB_PINGPONG
using RecordKernel = rocm_wmma_gemm::embedded_ab_pingpong_gemm;
#elif RECORD_PAIRED_A_HANDOFF
using RecordKernel = rocm_wmma_gemm::paired_a_handoff_gemm;
#elif RECORD_DIRECT_B_SHARED_A
using RecordKernel = rocm_wmma_gemm::direct_b_shared_a_gemm;
#elif RECORD_B_RESIDENT
using RecordKernel = rocm_wmma_gemm::b_resident_gemm;
#elif RECORD_BLOCK_512X128
using RecordKernel = rocm_wmma_gemm::block_512x128_gemm;
#elif RECORD_INTERLEAVED_PINGPONG
using RecordKernel = rocm_wmma_gemm::interleaved_pingpong_gemm;
#elif RECORD_PERSISTENT_BLOCK
using RecordKernel = rocm_wmma_gemm::persistent_block_gemm;
#elif RECORD_DIRECT_A_SHARED_B
using RecordKernel = rocm_wmma_gemm::direct_a_shared_b_gemm;
#elif RECORD_WAVEPRIVATE_A
using RecordKernel = rocm_wmma_gemm::waveprivate_a_gemm;
#elif RECORD_BLOCK_K2_RING
using RecordKernel = rocm_wmma_gemm::block_k2_ring_gemm;
#elif RECORD_BLOCK_PREPACKED
using RecordKernel = rocm_wmma_gemm::block_prepacked_gemm;
#elif RECORD_DIRECT_PREPACKED
using RecordKernel = rocm_wmma_gemm::prepacked_direct_gemm;
#elif RECORD_PRODUCER_CONSUMER
using RecordKernel = rocm_wmma_gemm::producer_consumer_gemm;
#else
using RecordKernel = rocm_wmma_gemm::kernel_gemm_impl<
    half,
    half,
    rocm_wmma_gemm::m_layout::row_major,
    RECORD_PREPACK_A ? rocm_wmma_gemm::m_layout::row_major
                     : rocm_wmma_gemm::m_layout::col_major,
    RECORD_PREPACK_B ? rocm_wmma_gemm::m_layout::col_major
                     : rocm_wmma_gemm::m_layout::row_major,
    RECORD_WARPS_M,
    RECORD_WARPS_N,
    RECORD_WARP_TILE_M,
    RECORD_WARP_TILE_N,
    RECORD_K_SLICES,
    RECORD_SINGLE_BUFFER,
    RECORD_SWIZZLE,
    RECORD_BITS,
    1>;
#endif

constexpr int kBlockM = RECORD_WARPS_M * RECORD_WARP_TILE_M * rocm_wmma_gemm::wmma_tile;
constexpr int kBlockN = RECORD_WARPS_N * RECORD_WARP_TILE_N * rocm_wmma_gemm::wmma_tile;
constexpr int kBlockK = RECORD_K_SLICES * rocm_wmma_gemm::wmma_tile;
constexpr int kThreads = RECORD_PRODUCER_CONSUMER
    ? RECORD_PRODUCER_WAVES * rocm_wmma_gemm::warp_size
    : rocm_wmma_gemm::warp_size * RECORD_WARPS_M * RECORD_WARPS_N;
static_assert(RECORD_PADDED_BLOCK_CONTRACT
                  || !RECORD_BLOCK_CONTRACT
                  || (kM % kBlockM == 0 && kN % kBlockN == 0
                      && kK % kBlockK == 0),
              "block-contract dimensions must be exact tile multiples");
static_assert(!RECORD_PADDED_BLOCK_CONTRACT || RECORD_BLOCK_CONTRACT,
              "padded edges apply only to a block-prepacked contract");
static_assert(!RECORD_PADDED_BLOCK_CONTRACT || kK % kBlockK == 0,
              "padded edges do not pad the reduction dimension");
static_assert(RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER >= 1,
              "B tile stride multiplier must be positive");
static_assert(RECORD_BLOCK_CONTRACT
                  || RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER == 1,
              "B tile stride padding requires a block/K contract");

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() % 2) != 0) {
        return values[middle];
    }
    return (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0;
}

bool finite_float(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x7f800000u) != 0x7f800000u;
}

float next_input(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    const int centered = static_cast<int>(state & 0x1fffu) - 4096;
    return static_cast<float>(centered) / 8192.0f;
}

}  // namespace

int main(int argc, char** argv) {
    const int warmup = argc > 1 ? std::atoi(argv[1]) : kWarmupDefault;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : kIterationsDefault;
    const size_t offset_a = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;
    const size_t offset_b = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 0;
    const size_t offset_c = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 0;
    const bool combined_ab = argc > 6 && std::strcmp(argv[6], "combined-ab") == 0;
    if (warmup < 1 || iterations < 1 || argc > 7
        || (argc > 6 && !combined_ab)) {
        std::fprintf(stderr,
                     "usage: %s [warmup] [iterations] [offset-a] [offset-b] "
                     "[offset-c] [combined-ab]\n",
                     argv[0]);
        return 2;
    }

    constexpr size_t a_elements = static_cast<size_t>(kM) * kK;
    constexpr size_t b_elements = static_cast<size_t>(kN) * kK;
    constexpr size_t c_elements = static_cast<size_t>(kM) * kN;
    std::vector<half> host_a(a_elements);
    std::vector<half> host_b(b_elements);
    uint32_t state_a = 0x243f6a88u;
    uint32_t state_b = 0x85a308d3u;
    for (size_t index = 0; index < a_elements; ++index) {
        host_a[index] = __float2half(next_input(state_a));
    }
    for (size_t index = 0; index < b_elements; ++index) {
        host_b[index] = __float2half(next_input(state_b));
    }
    std::vector<half> original_a;
    std::vector<half> original_b;
    if constexpr (RECORD_BLOCK_CONTRACT) {
        constexpr int block_m = kBlockM;
        constexpr int block_n = kBlockN;
        constexpr int block_k = kBlockK;
        constexpr int m_blocks = (kM + block_m - 1) / block_m;
        constexpr int n_blocks = (kN + block_n - 1) / block_n;
        original_a = host_a;
        original_b = host_b;
        std::vector<half> packed_a(
            static_cast<size_t>(m_blocks) * block_m * kK);
        std::vector<half> packed_b(
            static_cast<size_t>(n_blocks) * (kK / block_k)
            * block_n * block_k
            * RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER);
        for (int bm = 0; bm < m_blocks; ++bm) {
            for (int bk = 0; bk < kK / block_k; ++bk) {
                for (int m = 0; m < block_m; ++m) {
                    for (int k = 0; k < block_k; ++k) {
                        const size_t packed = RECORD_BLOCK_SLICE_MAJOR
                            ? ((((static_cast<size_t>(bm) * (kK / block_k)
                                      + bk)
                                     * (block_k / rocm_wmma_gemm::wmma_tile)
                                 + k / rocm_wmma_gemm::wmma_tile)
                                    * block_m
                                + m)
                                   * rocm_wmma_gemm::wmma_tile
                               + k % rocm_wmma_gemm::wmma_tile)
                            : (((static_cast<size_t>(bm) * (kK / block_k)
                                    + bk)
                                   * block_m
                               + m)
                                  * block_k
                              + k);
                        const int logical_m = bm * block_m + m;
                        if (logical_m < kM) {
                            const size_t logical =
                                static_cast<size_t>(bk * block_k + k) * kM
                                + logical_m;
                            packed_a[packed] = host_a[logical];
                        } else {
                            packed_a[packed] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        for (int bn = 0; bn < n_blocks; ++bn) {
            for (int bk = 0; bk < kK / block_k; ++bk) {
                for (int n = 0; n < block_n; ++n) {
                    for (int k = 0; k < block_k; ++k) {
                        const size_t tile_base =
                            (static_cast<size_t>(bn) * (kK / block_k) + bk)
                            * block_n * block_k
                            * RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER;
                        const size_t packed = tile_base
                            + (RECORD_BLOCK_SLICE_MAJOR
                                   ? ((static_cast<size_t>(
                                           k / rocm_wmma_gemm::wmma_tile)
                                           * block_n
                                       + n)
                                          * rocm_wmma_gemm::wmma_tile
                                      + k % rocm_wmma_gemm::wmma_tile)
                                   : static_cast<size_t>(n) * block_k + k);
                        const int logical_n = bn * block_n + n;
                        if (logical_n < kN) {
                            const size_t logical =
                                static_cast<size_t>(bk * block_k + k) * kN
                                + logical_n;
                            packed_b[packed] = host_b[logical];
                        } else {
                            packed_b[packed] = __float2half(0.0f);
                        }
                    }
                }
            }
        }
        host_a.swap(packed_a);
        host_b.swap(packed_b);
    }
    if constexpr (RECORD_PREPACK_A) {
        std::vector<half> packed_a(a_elements);
        for (int row = 0; row < kM; ++row) {
            for (int col = 0; col < kK; ++col) {
                packed_a[static_cast<size_t>(row) * kK + col]
                    = host_a[static_cast<size_t>(col) * kM + row];
            }
        }
        host_a.swap(packed_a);
    }
    if constexpr (RECORD_PREPACK_B) {
        std::vector<half> packed_b(b_elements);
        for (int row = 0; row < kK; ++row) {
            for (int col = 0; col < kN; ++col) {
                packed_b[static_cast<size_t>(col) * kK + row]
                    = host_b[static_cast<size_t>(row) * kN + col];
            }
        }
        host_b.swap(packed_b);
    }

    half* allocation_a = nullptr;
    half* allocation_b = nullptr;
    half* allocation_ab = nullptr;
    half* allocation_c = nullptr;
    half* device_reference = nullptr;
    half* device_a = nullptr;
    half* device_b = nullptr;
    const size_t a_storage_elements = host_a.size();
    const size_t b_storage_elements = host_b.size();
    if (combined_ab) {
        HIP_CHECK(hipMalloc(&allocation_ab,
                            (a_storage_elements + b_storage_elements
                             + offset_a + offset_b)
                                * sizeof(*allocation_ab)));
        device_a = allocation_ab + offset_a;
        device_b = device_a + a_storage_elements + offset_b;
    } else {
        HIP_CHECK(hipMalloc(&allocation_a,
                            (a_storage_elements + offset_a)
                                * sizeof(*allocation_a)));
        HIP_CHECK(hipMalloc(&allocation_b,
                            (b_storage_elements + offset_b)
                                * sizeof(*allocation_b)));
        device_a = allocation_a + offset_a;
        device_b = allocation_b + offset_b;
    }
    HIP_CHECK(hipMalloc(&allocation_c,
                        (c_elements + offset_c) * sizeof(*allocation_c)));
    half* const device_c = allocation_c + offset_c;
    HIP_CHECK(hipMalloc(&device_reference,
                        c_elements * sizeof(*device_reference)));
    HIP_CHECK(hipMemcpy(device_a, host_a.data(),
                        a_storage_elements * sizeof(*device_a),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(),
                        b_storage_elements * sizeof(*device_b),
                        hipMemcpyHostToDevice));

    // CPU and GPU share the Strix Halo package power envelope. Let the CPU-side
    // deterministic input generation settle before beginning GPU warm-up, so
    // host preparation cannot depress the measured GPU clock window.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
    hipStream_t stream_b = nullptr;
    HIP_CHECK(hipStreamCreate(&stream_b));
#endif
    const int grid_m = (kM + kBlockM - 1) / kBlockM;
    const int grid_n = (kN + kBlockN - 1) / kBlockN;
    const dim3 grid(RECORD_GRID_BLOCKS > 0
                        ? RECORD_GRID_BLOCKS
                        : grid_m * grid_n);
#if RECORD_SPLIT_MODULE_DISPATCH
    const dim3 grid_b(RECORD_GRID_BLOCKS_B);
#if RECORD_SPLIT_CONCURRENT
    constexpr const char* dispatch_name = "split-two-stream";
#else
    constexpr const char* dispatch_name = "split-serial";
#endif
#else
    constexpr const char* dispatch_name = "single";
#endif
    const dim3 block(kThreads);
    int active_blocks_per_cu = 0;
#if RECORD_SPLIT_MODULE_DISPATCH
    static_assert(RECORD_GRID_BLOCKS > 0,
                  "split module dispatch needs an explicit half-grid size");
    static_assert(RECORD_GRID_BLOCKS + RECORD_GRID_BLOCKS_B
                      == (kM / kBlockM) * (kN / kBlockN),
                  "split module grids must cover the complete output once");
    hipModule_t module0 = nullptr;
    hipModule_t module1 = nullptr;
    hipFunction_t function0 = nullptr;
    hipFunction_t function1 = nullptr;
    constexpr const char* kernel_symbol =
        "_ZN14rocm_wmma_gemm20block_prepacked_gemm3runEP6__halfPKS1_S4_iii";
    HIP_CHECK(hipModuleLoad(&module0, RECORD_MODULE0_PATH));
    HIP_CHECK(hipModuleLoad(&module1, RECORD_MODULE1_PATH));
    HIP_CHECK(hipModuleGetFunction(&function0, module0, kernel_symbol));
    HIP_CHECK(hipModuleGetFunction(&function1, module1, kernel_symbol));
    int active_blocks_module1 = 0;
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_cu, function0, kThreads, RECORD_DYNAMIC_LDS_BYTES));
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_module1, function1, kThreads,
        RECORD_DYNAMIC_LDS_BYTES));
    if (active_blocks_per_cu != active_blocks_module1) {
        std::fprintf(stderr, "split module occupancy mismatch: %d != %d\n",
                     active_blocks_per_cu, active_blocks_module1);
        return 2;
    }
#else
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_cu,
        RecordKernel::run,
        kThreads,
        RECORD_DYNAMIC_LDS_BYTES));
#endif
    std::printf("occupancy active_blocks_per_cu=%d active_waves_per_cu=%d\n",
                active_blocks_per_cu,
                active_blocks_per_cu * kThreads / rocm_wmma_gemm::warp_size);

    auto launch = [&]() {
#if RECORD_SPLIT_MODULE_DISPATCH
        half* output = device_c;
        const half* input_a = device_a;
        const half* input_b = device_b;
        int m = kM;
        int n = kN;
        int k = kK;
        void* kernel_params[] = {
            &output, &input_a, &input_b, &m, &n, &k,
        };
        HIP_CHECK(hipModuleLaunchKernel(
            function0, grid.x, 1, 1, block.x, 1, 1,
            RECORD_DYNAMIC_LDS_BYTES, stream, kernel_params, nullptr));
        HIP_CHECK(hipModuleLaunchKernel(
            function1, grid_b.x, 1, 1, block.x, 1, 1,
            RECORD_DYNAMIC_LDS_BYTES,
#if RECORD_SPLIT_CONCURRENT
            stream_b,
#else
            stream,
#endif
            kernel_params, nullptr));
#else
        hipLaunchKernelGGL(RecordKernel::run,
                           grid,
                           block,
                           RECORD_DYNAMIC_LDS_BYTES,
                           stream,
                           device_c, device_a, device_b,
                           kM, kN, kK);
        HIP_CHECK(hipPeekAtLastError());
#endif
    };

    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    HIP_CHECK(hipStreamSynchronize(stream));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
    HIP_CHECK(hipStreamSynchronize(stream_b));
#endif

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
    hipEvent_t stop_b = nullptr;
    HIP_CHECK(hipEventCreate(&stop_b));
#endif
    std::vector<float> times_ms;
    times_ms.reserve(kTimingBlocks);
    for (int timing_block = 0; timing_block < kTimingBlocks; ++timing_block) {
        HIP_CHECK(hipEventRecord(start, stream));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
        // Both queues measure from the same device-side epoch. The slower stop
        // is the completion time for the full output, not the sum of streams.
        HIP_CHECK(hipStreamWaitEvent(stream_b, start, 0));
#endif
        for (int index = 0; index < iterations; ++index) {
            launch();
        }
        HIP_CHECK(hipEventRecord(stop, stream));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
        HIP_CHECK(hipEventRecord(stop_b, stream_b));
#endif
        HIP_CHECK(hipEventSynchronize(stop));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
        HIP_CHECK(hipEventSynchronize(stop_b));
#endif
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
        float elapsed_b_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_b_ms, start, stop_b));
        elapsed_ms = std::max(elapsed_ms, elapsed_b_ms);
#endif
        times_ms.push_back(elapsed_ms / iterations);
    }

    const double median_ms = median(times_ms);
    const double operations = 2.0 * kM * kN * static_cast<double>(kK);
    const double tflops = operations / (median_ms * 1.0e-3) / 1.0e12;

    if constexpr (RECORD_BLOCK_CONTRACT) {
        // rocBLAS has no descriptor for the block/K-major contract. Restore
        // the original matrices after timing so the independent reference
        // still uses its ordinary column-major-A/row-major-B interface.
        HIP_CHECK(hipMemcpy(device_a, original_a.data(),
                            a_elements * sizeof(*device_a), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_b, original_b.data(),
                            b_elements * sizeof(*device_b), hipMemcpyHostToDevice));
    }

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // rocBLAS is column-major. A is already column-major. Row-major B storage
    // represents column-major B^T, and row-major C storage represents
    // column-major C^T. Therefore compute C^T = B^T * A^T.
    const rocblas_operation op_b = RECORD_PREPACK_B
        ? rocblas_operation_transpose : rocblas_operation_none;
    const rocblas_operation op_a = RECORD_PREPACK_A
        ? rocblas_operation_none : rocblas_operation_transpose;
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, op_b, op_a,
        kN, kM, kK, &alpha,
        device_b, rocblas_datatype_f16_r, kN,
        device_a, rocblas_datatype_f16_r, kM,
        &beta,
        device_reference, rocblas_datatype_f16_r, kN,
        device_reference, rocblas_datatype_f16_r, kN,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<half> host_c(c_elements);
    std::vector<half> host_reference(c_elements);
    HIP_CHECK(hipMemcpy(host_c.data(), device_c, c_elements * sizeof(half),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_reference.data(), device_reference,
                        c_elements * sizeof(half), hipMemcpyDeviceToHost));

    bool finite = true;
    double max_abs_error = 0.0;
    double reference_max_abs = 0.0;
    double squared_error = 0.0;
    double dot_product = 0.0;
    double norm_actual = 0.0;
    double norm_reference = 0.0;
    for (size_t index = 0; index < c_elements; ++index) {
        const float actual_float = __half2float(host_c[index]);
        const float expected_float = __half2float(host_reference[index]);
        const double actual = actual_float;
        const double expected = expected_float;
        const double difference = actual - expected;
        finite = finite && finite_float(actual_float)
            && finite_float(expected_float);
        max_abs_error = std::max(max_abs_error, std::fabs(difference));
        reference_max_abs = std::max(reference_max_abs, std::fabs(expected));
        squared_error += difference * difference;
        dot_product += actual * expected;
        norm_actual += actual * actual;
        norm_reference += expected * expected;
    }
    const double normalized_max_error = max_abs_error
        / std::max(reference_max_abs, 1.0);
    const double rms_error = std::sqrt(squared_error / c_elements);
    const double cosine_similarity = dot_product
        / std::sqrt(norm_actual * norm_reference);
    const bool correct = finite && cosine_similarity > 0.995
                         && normalized_max_error < 0.03;
    const bool promoted = correct && tflops > kPromotionTargetTflops;

    std::printf(
        "config=%d,%d,%d,%d,%d,%d,%d,%d shape=%dx%dx%d block=%dx%d k=%d cu_mode=1 "
        "warmup=%d timing_blocks=%d iterations_per_block=%d offsets=%zu,%zu,%zu "
        "allocation=%s input_contract=%s pipeline=%s dispatch=%s grids=%u,%u\n",
        RECORD_WARPS_M, RECORD_WARPS_N, RECORD_WARP_TILE_M, RECORD_WARP_TILE_N,
        RECORD_K_SLICES, RECORD_SINGLE_BUFFER, RECORD_SWIZZLE, RECORD_BITS,
        kM, kN, kK, kBlockM, kBlockN, kBlockK, warmup, kTimingBlocks, iterations,
        offset_a, offset_b, offset_c, combined_ab ? "combined-ab" : "separate",
        RECORD_BLOCK_CONTRACT
            ? (RECORD_BLOCK_B_TILE_STRIDE_MULTIPLIER > 1
                   ? "A/B-block-K16-prepacked-B-stride-padded"
                   : RECORD_PADDED_BLOCK_CONTRACT
                   ? "A/B-block-K16-prepacked-zero-padded-edges"
                   : RECORD_BLOCK_SLICE_MAJOR
                   ? "A/B-block-K32-slice-prepacked"
                   : "A/B-block-K16-prepacked")
            : (RECORD_PREPACK_A ? (RECORD_PREPACK_B ? "A-row/B-col-prepacked"
                                                        : "A-row-prepacked/B-row")
                                : (RECORD_PREPACK_B ? "A-col/B-col-prepacked"
                                                    : "A-col/B-row-original")),
        RECORD_COMPACT_XOR_PINGPONG
            ? "compact-XOR-pingpong"
            : RECORD_BLOCK_K32_PUBLICATION
            ? "blocked-K32-publication"
            : RECORD_EMBEDDED_AB_PINGPONG
            ? "embedded-padding-ab-pingpong"
            : RECORD_BLOCK_CONTRACT
            ? "blocked-native-fragment"
            : (RECORD_PRODUCER_CONSUMER ? "inter-wave-producer-consumer"
                                        : "cooperative"),
        dispatch_name, grid.x,
#if RECORD_SPLIT_MODULE_DISPATCH
        grid_b.x
#else
        0u
#endif
    );
    std::printf(
        "median_ms=%.6f tflops=%.6f min_ms=%.6f max_ms=%.6f\n",
        median_ms, tflops,
        static_cast<double>(*std::min_element(times_ms.begin(), times_ms.end())),
        static_cast<double>(*std::max_element(times_ms.begin(), times_ms.end())));
    std::printf(
        "validation finite=%s elements=%zu max_abs_error=%.9f "
        "reference_max_abs=%.9f normalized_max_error=%.9f rms_error=%.9f "
        "cosine_similarity=%.9f\n",
        finite ? "yes" : "no", c_elements, max_abs_error, reference_max_abs,
        normalized_max_error, rms_error, cosine_similarity);
    std::printf("promotion target_tflops=%.3f pass=%s\n",
                kPromotionTargetTflops, promoted ? "yes" : "no");

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
    HIP_CHECK(hipEventDestroy(stop_b));
#endif
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
#if RECORD_SPLIT_MODULE_DISPATCH
    HIP_CHECK(hipModuleUnload(module1));
    HIP_CHECK(hipModuleUnload(module0));
#endif
#if RECORD_SPLIT_MODULE_DISPATCH && RECORD_SPLIT_CONCURRENT
    HIP_CHECK(hipStreamDestroy(stream_b));
#endif
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(device_reference));
    HIP_CHECK(hipFree(allocation_c));
    if (combined_ab) {
        HIP_CHECK(hipFree(allocation_ab));
    } else {
        HIP_CHECK(hipFree(allocation_b));
        HIP_CHECK(hipFree(allocation_a));
    }
    return promoted ? 0 : 1;
}
