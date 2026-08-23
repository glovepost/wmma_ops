// Reproducible standalone harness for the promoted gfx1151 record kernel.
//
// Kernel implementation: adelj88/rocm_wmma_gemm at
// 281b5dfd7fbff9cea80753bc55274a54f4a7c53a (MIT).
// This file supplies the exact tuned configuration, timing protocol, and an
// independent full-output rocBLAS reference. Compile through
// tools/build_rocwmma_record.sh so CU mode and the pinned include tree are not
// accidentally omitted.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocm_wmma_gemm/kernel/kernel.hpp>

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

constexpr int kSize = 4096;
constexpr int kWarmupDefault = 200;
constexpr int kIterationsDefault = 100;
constexpr int kTimingBlocks = 5;
constexpr double kPromotionTargetTflops = 41.0;

// Tuned parameter order:
// warps_m, warps_n, warp_tile_m, warp_tile_n, k_slices, single_buffer,
// swizzle, vector bits, alignment fast path.
using RecordKernel = rocm_wmma_gemm::kernel_gemm_impl<
    float,
    half,
    rocm_wmma_gemm::m_layout::row_major,
    rocm_wmma_gemm::m_layout::row_major,
    rocm_wmma_gemm::m_layout::row_major,
    1,
    8,
    7,
    2,
    2,
    1,
    8,
    128,
    0>;

constexpr int kBlockM = 1 * 7 * rocm_wmma_gemm::wmma_tile;
constexpr int kBlockN = 8 * 2 * rocm_wmma_gemm::wmma_tile;
constexpr int kThreads = rocm_wmma_gemm::warp_size * 1 * 8;

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

    half* allocation_a = nullptr;
    half* allocation_b = nullptr;
    half* allocation_ab = nullptr;
    float* allocation_c = nullptr;
    float* device_reference = nullptr;
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
    float* const device_c = allocation_c + offset_c;
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
    const int grid_n = kSize / kBlockN;
    const dim3 grid(grid_m * grid_n);
    const dim3 block(kThreads);

    auto launch = [&]() {
        hipLaunchKernelGGL(RecordKernel::run, grid, block, 0, stream,
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

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // rocBLAS is column-major. Row-major C=A*B is the equivalent
    // column-major C^T=B^T*A^T operation over the same storage.
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle, rocblas_operation_none, rocblas_operation_none,
        kSize, kSize, kSize, &alpha,
        device_b, rocblas_datatype_f16_r, kSize,
        device_a, rocblas_datatype_f16_r, kSize,
        &beta,
        device_reference, rocblas_datatype_f32_r, kSize,
        device_reference, rocblas_datatype_f32_r, kSize,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> host_c(elements);
    std::vector<float> host_reference(elements);
    HIP_CHECK(hipMemcpy(host_c.data(), device_c, elements * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_reference.data(), device_reference,
                        elements * sizeof(float), hipMemcpyDeviceToHost));

    bool finite = true;
    double max_abs_error = 0.0;
    double reference_max_abs = 0.0;
    double squared_error = 0.0;
    for (size_t index = 0; index < elements; ++index) {
        const float actual_float = host_c[index];
        const float expected_float = host_reference[index];
        const double actual = actual_float;
        const double expected = expected_float;
        const double difference = actual - expected;
        finite = finite && finite_float(actual_float)
            && finite_float(expected_float);
        max_abs_error = std::max(max_abs_error, std::fabs(difference));
        reference_max_abs = std::max(reference_max_abs, std::fabs(expected));
        squared_error += difference * difference;
    }
    const double normalized_max_error = max_abs_error
        / std::max(reference_max_abs, 1.0);
    const double rms_error = std::sqrt(squared_error / elements);
    const bool correct = finite && normalized_max_error < 0.01;
    const bool promoted = correct && tflops > kPromotionTargetTflops;

    std::printf(
        "config=1,8,7,2,2,1,8,128 block=%dx%d k=32 cu_mode=1 "
        "warmup=%d timing_blocks=%d iterations_per_block=%d offsets=%zu,%zu,%zu "
        "allocation=%s\n",
        kBlockM, kBlockN, warmup, kTimingBlocks, iterations,
        offset_a, offset_b, offset_c, combined_ab ? "combined-ab" : "separate");
    std::printf(
        "median_ms=%.6f tflops=%.6f min_ms=%.6f max_ms=%.6f\n",
        median_ms, tflops,
        static_cast<double>(*std::min_element(times_ms.begin(), times_ms.end())),
        static_cast<double>(*std::max_element(times_ms.begin(), times_ms.end())));
    std::printf(
        "validation finite=%s elements=%zu max_abs_error=%.9f "
        "reference_max_abs=%.9f normalized_max_error=%.9f rms_error=%.9f\n",
        finite ? "yes" : "no", elements, max_abs_error, reference_max_abs,
        normalized_max_error, rms_error);
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
