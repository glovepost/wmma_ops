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
#ifndef RECORD_BLOCK_SLICE_MAJOR
#define RECORD_BLOCK_SLICE_MAJOR 0
#endif
#ifndef RECORD_PRODUCER_WAVES
#define RECORD_PRODUCER_WAVES 5
#endif
#define RECORD_BLOCK_CONTRACT \
    (RECORD_BLOCK_PREPACKED | RECORD_BLOCK_K2_RING)

constexpr int kSize = 4096;
constexpr int kWarmupDefault = 200;
constexpr int kIterationsDefault = 100;
constexpr int kTimingBlocks = 5;
constexpr double kPromotionTargetTflops = 50.0;

// Tuned parameter order:
// warps_m, warps_n, warp_tile_m, warp_tile_n, k_slices, single_buffer,
// swizzle, vector bits, alignment fast path.
#if RECORD_BLOCK_K2_RING
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

    constexpr size_t elements = static_cast<size_t>(kSize) * kSize;
    std::vector<half> host_a(elements);
    std::vector<half> host_b(elements);
    uint32_t state_a = 0x243f6a88u;
    uint32_t state_b = 0x85a308d3u;
    for (size_t index = 0; index < elements; ++index) {
        host_a[index] = __float2half(next_input(state_a));
        host_b[index] = __float2half(next_input(state_b));
    }
    std::vector<half> original_a;
    std::vector<half> original_b;
    if constexpr (RECORD_BLOCK_CONTRACT) {
        constexpr int block_m = kBlockM;
        constexpr int block_n = kBlockN;
        constexpr int block_k = kBlockK;
        original_a = host_a;
        original_b = host_b;
        std::vector<half> packed_a(elements);
        std::vector<half> packed_b(elements);
        for (int bm = 0; bm < kSize / block_m; ++bm) {
            for (int bk = 0; bk < kSize / block_k; ++bk) {
                for (int m = 0; m < block_m; ++m) {
                    for (int k = 0; k < block_k; ++k) {
                        const size_t packed = RECORD_BLOCK_SLICE_MAJOR
                            ? ((((static_cast<size_t>(bm) * (kSize / block_k)
                                      + bk)
                                     * (block_k / rocm_wmma_gemm::wmma_tile)
                                 + k / rocm_wmma_gemm::wmma_tile)
                                    * block_m
                                + m)
                                   * rocm_wmma_gemm::wmma_tile
                               + k % rocm_wmma_gemm::wmma_tile)
                            : (((static_cast<size_t>(bm) * (kSize / block_k)
                                    + bk)
                                   * block_m
                               + m)
                                  * block_k
                              + k);
                        const size_t logical =
                            static_cast<size_t>(bk * block_k + k) * kSize
                            + bm * block_m + m;
                        packed_a[packed] = host_a[logical];
                    }
                }
            }
        }
        for (int bn = 0; bn < kSize / block_n; ++bn) {
            for (int bk = 0; bk < kSize / block_k; ++bk) {
                for (int n = 0; n < block_n; ++n) {
                    for (int k = 0; k < block_k; ++k) {
                        const size_t packed = RECORD_BLOCK_SLICE_MAJOR
                            ? ((((static_cast<size_t>(bn) * (kSize / block_k)
                                      + bk)
                                     * (block_k / rocm_wmma_gemm::wmma_tile)
                                 + k / rocm_wmma_gemm::wmma_tile)
                                    * block_n
                                + n)
                                   * rocm_wmma_gemm::wmma_tile
                               + k % rocm_wmma_gemm::wmma_tile)
                            : (((static_cast<size_t>(bn) * (kSize / block_k)
                                    + bk)
                                   * block_n
                               + n)
                                  * block_k
                              + k);
                        const size_t logical =
                            static_cast<size_t>(bk * block_k + k) * kSize
                            + bn * block_n + n;
                        packed_b[packed] = host_b[logical];
                    }
                }
            }
        }
        host_a.swap(packed_a);
        host_b.swap(packed_b);
    }
    if constexpr (RECORD_PREPACK_A) {
        std::vector<half> packed_a(elements);
        for (int row = 0; row < kSize; ++row) {
            for (int col = 0; col < kSize; ++col) {
                packed_a[static_cast<size_t>(row) * kSize + col]
                    = host_a[static_cast<size_t>(col) * kSize + row];
            }
        }
        host_a.swap(packed_a);
    }
    if constexpr (RECORD_PREPACK_B) {
        std::vector<half> packed_b(elements);
        for (int row = 0; row < kSize; ++row) {
            for (int col = 0; col < kSize; ++col) {
                packed_b[static_cast<size_t>(col) * kSize + row]
                    = host_b[static_cast<size_t>(row) * kSize + col];
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
    if (combined_ab) {
        HIP_CHECK(hipMalloc(&allocation_ab,
                            (2 * elements + offset_a + offset_b)
                                * sizeof(*allocation_ab)));
        device_a = allocation_ab + offset_a;
        device_b = device_a + elements + offset_b;
    } else {
        HIP_CHECK(hipMalloc(&allocation_a,
                            (elements + offset_a) * sizeof(*allocation_a)));
        HIP_CHECK(hipMalloc(&allocation_b,
                            (elements + offset_b) * sizeof(*allocation_b)));
        device_a = allocation_a + offset_a;
        device_b = allocation_b + offset_b;
    }
    HIP_CHECK(hipMalloc(&allocation_c,
                        (elements + offset_c) * sizeof(*allocation_c)));
    half* const device_c = allocation_c + offset_c;
    HIP_CHECK(hipMalloc(&device_reference,
                        elements * sizeof(*device_reference)));
    HIP_CHECK(hipMemcpy(device_a, host_a.data(), elements * sizeof(*device_a),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), elements * sizeof(*device_b),
                        hipMemcpyHostToDevice));

    // CPU and GPU share the Strix Halo package power envelope. Let the CPU-side
    // deterministic input generation settle before beginning GPU warm-up, so
    // host preparation cannot depress the measured GPU clock window.
    std::this_thread::sleep_for(std::chrono::seconds(2));

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    const int grid_m = (kSize + kBlockM - 1) / kBlockM;
    const int grid_n = (kSize + kBlockN - 1) / kBlockN;
    const dim3 grid(grid_m * grid_n);
    const dim3 block(kThreads);
    int active_blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_cu,
        RecordKernel::run,
        kThreads,
        RECORD_DYNAMIC_LDS_BYTES));
    std::printf("occupancy active_blocks_per_cu=%d active_waves_per_cu=%d\n",
                active_blocks_per_cu,
                active_blocks_per_cu * kThreads / rocm_wmma_gemm::warp_size);

    auto launch = [&]() {
        hipLaunchKernelGGL(RecordKernel::run,
                           grid,
                           block,
                           RECORD_DYNAMIC_LDS_BYTES,
                           stream,
                           device_c, device_a, device_b,
                           kSize, kSize, kSize);
        HIP_CHECK(hipPeekAtLastError());
    };

    for (int index = 0; index < warmup; ++index) {
        launch();
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> times_ms;
    times_ms.reserve(kTimingBlocks);
    for (int timing_block = 0; timing_block < kTimingBlocks; ++timing_block) {
        HIP_CHECK(hipEventRecord(start, stream));
        for (int index = 0; index < iterations; ++index) {
            launch();
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        times_ms.push_back(elapsed_ms / iterations);
    }

    const double median_ms = median(times_ms);
    const double operations = 2.0 * kSize * kSize * static_cast<double>(kSize);
    const double tflops = operations / (median_ms * 1.0e-3) / 1.0e12;

    if constexpr (RECORD_BLOCK_CONTRACT) {
        // rocBLAS has no descriptor for the block/K-major contract. Restore
        // the original matrices after timing so the independent reference
        // still uses its ordinary column-major-A/row-major-B interface.
        HIP_CHECK(hipMemcpy(device_a, original_a.data(),
                            elements * sizeof(*device_a), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(device_b, original_b.data(),
                            elements * sizeof(*device_b), hipMemcpyHostToDevice));
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
        kSize, kSize, kSize, &alpha,
        device_b, rocblas_datatype_f16_r, kSize,
        device_a, rocblas_datatype_f16_r, kSize,
        &beta,
        device_reference, rocblas_datatype_f16_r, kSize,
        device_reference, rocblas_datatype_f16_r, kSize,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<half> host_c(elements);
    std::vector<half> host_reference(elements);
    HIP_CHECK(hipMemcpy(host_c.data(), device_c, elements * sizeof(half),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_reference.data(), device_reference,
                        elements * sizeof(half), hipMemcpyDeviceToHost));

    bool finite = true;
    double max_abs_error = 0.0;
    double reference_max_abs = 0.0;
    double squared_error = 0.0;
    double dot_product = 0.0;
    double norm_actual = 0.0;
    double norm_reference = 0.0;
    for (size_t index = 0; index < elements; ++index) {
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
    const double rms_error = std::sqrt(squared_error / elements);
    const double cosine_similarity = dot_product
        / std::sqrt(norm_actual * norm_reference);
    const bool correct = finite && cosine_similarity > 0.995
                         && normalized_max_error < 0.03;
    const bool promoted = correct && tflops > kPromotionTargetTflops;

    std::printf(
        "config=%d,%d,%d,%d,%d,%d,%d,%d block=%dx%d k=%d cu_mode=1 "
        "warmup=%d timing_blocks=%d iterations_per_block=%d offsets=%zu,%zu,%zu "
        "allocation=%s input_contract=%s pipeline=%s\n",
        RECORD_WARPS_M, RECORD_WARPS_N, RECORD_WARP_TILE_M, RECORD_WARP_TILE_N,
        RECORD_K_SLICES, RECORD_SINGLE_BUFFER, RECORD_SWIZZLE, RECORD_BITS,
        kBlockM, kBlockN, kBlockK, warmup, kTimingBlocks, iterations,
        offset_a, offset_b, offset_c, combined_ab ? "combined-ab" : "separate",
        RECORD_BLOCK_CONTRACT
            ? (RECORD_BLOCK_SLICE_MAJOR
                   ? "A/B-block-K32-slice-prepacked"
                   : "A/B-block-K16-prepacked")
            : (RECORD_PREPACK_A ? (RECORD_PREPACK_B ? "A-row/B-col-prepacked"
                                                        : "A-row-prepacked/B-row")
                                : (RECORD_PREPACK_B ? "A-col/B-col-prepacked"
                                                    : "A-col/B-row-original")),
        RECORD_BLOCK_CONTRACT
            ? "blocked-native-fragment"
            : (RECORD_PRODUCER_CONSUMER ? "inter-wave-producer-consumer"
                                        : "cooperative"));
    std::printf(
        "median_ms=%.6f tflops=%.6f min_ms=%.6f max_ms=%.6f\n",
        median_ms, tflops,
        static_cast<double>(*std::min_element(times_ms.begin(), times_ms.end())),
        static_cast<double>(*std::max_element(times_ms.begin(), times_ms.end())));
    std::printf(
        "validation finite=%s elements=%zu max_abs_error=%.9f "
        "reference_max_abs=%.9f normalized_max_error=%.9f rms_error=%.9f "
        "cosine_similarity=%.9f\n",
        finite ? "yes" : "no", elements, max_abs_error, reference_max_abs,
        normalized_max_error, rms_error, cosine_similarity);
    std::printf("promotion target_tflops=%.3f pass=%s\n",
                kPromotionTargetTflops, promoted ? "yes" : "no");

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
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
