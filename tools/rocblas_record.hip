// Fair rocBLAS reference for the repository's 4096-cubed record contract:
// row-major FP16 A/B, FP32 accumulation, and FP32 output.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char** argv) {
    const int size = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 10;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 50;
    const bool fp16_output = argc > 4 && std::strcmp(argv[4], "f16") == 0;
    const bool tune_solutions = argc > 5 && std::strcmp(argv[5], "tune") == 0;
    const bool sweep_solutions = argc > 5
        && std::strcmp(argv[5], "sweep") == 0;
    const bool default_solution = argc > 5
        && std::strcmp(argv[5], "default") == 0;
    const bool explicit_solution = argc > 5 && !tune_solutions
        && !sweep_solutions
        && !default_solution;
    const int32_t requested_solution = explicit_solution
        ? static_cast<int32_t>(std::strtol(argv[5], nullptr, 10)) : 0;
    const int leading_dimension = argc > 6 ? std::atoi(argv[6]) : size;
    const size_t offset_a = argc > 7
        ? static_cast<size_t>(std::strtoull(argv[7], nullptr, 10)) : 0;
    const size_t offset_b = argc > 8
        ? static_cast<size_t>(std::strtoull(argv[8], nullptr, 10)) : 0;
    const size_t offset_c = argc > 9
        ? static_cast<size_t>(std::strtoull(argv[9], nullptr, 10)) : 0;
    if (size <= 0 || warmup < 0 || iterations <= 0
        || leading_dimension < size) {
        std::fprintf(stderr,
                     "usage: %s [size] [warmup] [iterations] "
                     "[f16|f32] [default|tune|sweep|solution] "
                     "[leading-dimension] [offset-a] [offset-b] [offset-c]\n",
                     argv[0]);
        return 2;
    }

    const size_t elements = static_cast<size_t>(size) * leading_dimension;
    std::vector<__half> host_a(elements);
    std::vector<__half> host_b(elements);
    for (size_t i = 0; i < elements; ++i) {
        host_a[i] = __float2half(static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f);
        host_b[i] = __float2half(static_cast<float>(static_cast<int>(i % 13) - 6) / 16.0f);
    }

    __half* allocation_a = nullptr;
    __half* allocation_b = nullptr;
    void* allocation_c = nullptr;
    HIP_CHECK(hipMalloc(&allocation_a,
                        (elements + offset_a) * sizeof(*allocation_a)));
    HIP_CHECK(hipMalloc(&allocation_b,
                        (elements + offset_b) * sizeof(*allocation_b)));
    __half* const device_a = allocation_a + offset_a;
    __half* const device_b = allocation_b + offset_b;
    const size_t output_bytes = elements
        * (fp16_output ? sizeof(__half) : sizeof(float));
    const size_t output_element_bytes = fp16_output
        ? sizeof(__half) : sizeof(float);
    HIP_CHECK(hipMalloc(&allocation_c,
                        output_bytes + offset_c * output_element_bytes));
    void* const device_c = fp16_output
        ? static_cast<void*>(static_cast<__half*>(allocation_c) + offset_c)
        : static_cast<void*>(static_cast<float*>(allocation_c) + offset_c);
    HIP_CHECK(hipMemcpy(device_a, host_a.data(), elements * sizeof(*device_a),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), elements * sizeof(*device_b),
                        hipMemcpyHostToDevice));

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // rocBLAS is column-major. Row-major C=A*B is equivalent to the
    // column-major operation C^T=B^T*A^T using the same storage.
    rocblas_gemm_algo selected_algo = rocblas_gemm_algo_standard;
    int32_t selected_solution = 0;
    if (explicit_solution) {
        selected_algo = rocblas_gemm_algo_solution_index;
        selected_solution = requested_solution;
        std::printf("using explicit rocBLAS solution %d\n", selected_solution);
    }
    if (offset_a || offset_b || offset_c) {
        std::printf("storage offsets: A=%zu, B=%zu, C=%zu elements\n",
                    offset_a, offset_b, offset_c);
    }
    auto gemm = [&]() {
        return rocblas_gemm_ex(
            handle, rocblas_operation_none, rocblas_operation_none,
            size, size, size, &alpha,
            device_b, rocblas_datatype_f16_r, leading_dimension,
            device_a, rocblas_datatype_f16_r, leading_dimension,
            &beta,
            device_c, fp16_output ? rocblas_datatype_f16_r
                                  : rocblas_datatype_f32_r, leading_dimension,
            device_c, fp16_output ? rocblas_datatype_f16_r
                                  : rocblas_datatype_f32_r, leading_dimension,
            rocblas_datatype_f32_r, selected_algo,
            selected_solution, 0);
    };

    if (tune_solutions || sweep_solutions) {
        rocblas_int solution_count = 0;
        std::vector<rocblas_int> solutions;
        if (tune_solutions) {
            ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions(
                handle, rocblas_operation_none, rocblas_operation_none,
                size, size, size, &alpha,
                device_b, rocblas_datatype_f16_r, leading_dimension,
                device_a, rocblas_datatype_f16_r, leading_dimension,
                &beta,
                device_c, fp16_output ? rocblas_datatype_f16_r
                                      : rocblas_datatype_f32_r,
                leading_dimension,
                device_c, fp16_output ? rocblas_datatype_f16_r
                                      : rocblas_datatype_f32_r,
                leading_dimension,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0,
                nullptr, &solution_count));
            solutions.resize(static_cast<size_t>(solution_count));
            ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions(
                handle, rocblas_operation_none, rocblas_operation_none,
                size, size, size, &alpha,
                device_b, rocblas_datatype_f16_r, leading_dimension,
                device_a, rocblas_datatype_f16_r, leading_dimension,
                &beta,
                device_c, fp16_output ? rocblas_datatype_f16_r
                                      : rocblas_datatype_f32_r,
                leading_dimension,
                device_c, fp16_output ? rocblas_datatype_f16_r
                                      : rocblas_datatype_f32_r,
                leading_dimension,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0,
                solutions.data(), &solution_count));
        } else {
            // Libraries generated by make_tensile_hss_variants.py number
            // their 100 AMD source schedules from -11 through -110. The
            // ordinary rocBLAS query returns only schedules selected by the
            // library's size table; sweep all of them to retune after changing
            // the output type from FP16 to FP32.
            constexpr rocblas_int generated_solution_count = 100;
            solutions.reserve(generated_solution_count);
            for (rocblas_int index = 0; index < generated_solution_count;
                 ++index) {
                solutions.push_back(-11 - index);
            }
            solution_count = generated_solution_count;
        }

        struct TimedSolution {
            int32_t index;
            double milliseconds;
        };
        std::vector<TimedSolution> timings;
        selected_algo = rocblas_gemm_algo_solution_index;
        if (sweep_solutions && warmup > 0) {
            for (const rocblas_int index : solutions) {
                selected_solution = index;
                if (gemm() != rocblas_status_success) {
                    continue;
                }
                for (int i = 1; i < warmup; ++i) {
                    ROCBLAS_CHECK(gemm());
                }
                HIP_CHECK(hipDeviceSynchronize());
                break;
            }
        }
        for (const rocblas_int index : solutions) {
            selected_solution = index;
            if (gemm() != rocblas_status_success) {
                continue;
            }
            HIP_CHECK(hipDeviceSynchronize());
            hipEvent_t tune_start = nullptr;
            hipEvent_t tune_stop = nullptr;
            HIP_CHECK(hipEventCreate(&tune_start));
            HIP_CHECK(hipEventCreate(&tune_stop));
            HIP_CHECK(hipEventRecord(tune_start));
            constexpr int tune_iterations = 3;
            bool valid = true;
            for (int i = 0; i < tune_iterations; ++i) {
                if (gemm() != rocblas_status_success) {
                    valid = false;
                    break;
                }
            }
            HIP_CHECK(hipEventRecord(tune_stop));
            HIP_CHECK(hipEventSynchronize(tune_stop));
            float tune_elapsed_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(
                &tune_elapsed_ms, tune_start, tune_stop));
            HIP_CHECK(hipEventDestroy(tune_stop));
            HIP_CHECK(hipEventDestroy(tune_start));
            if (valid) {
                timings.push_back({
                    index,
                    static_cast<double>(tune_elapsed_ms) / tune_iterations,
                });
            }
        }
        std::sort(timings.begin(), timings.end(),
                  [](const TimedSolution& left, const TimedSolution& right) {
                      return left.milliseconds < right.milliseconds;
                  });
        std::printf(tune_solutions
                        ? "rocBLAS reported %d compatible solutions\n"
                        : "swept %d generated solutions\n",
                    solution_count);
        const size_t print_count = sweep_solutions
            ? timings.size() : std::min<size_t>(10, timings.size());
        for (size_t i = 0; i < print_count; ++i) {
            const double operations =
                2.0 * size * size * static_cast<double>(size);
            const double tflops = operations
                / (timings[i].milliseconds * 1.0e-3) / 1.0e12;
            std::printf("solution %d: %.6f ms, %.3f TFLOPS\n",
                        timings[i].index, timings[i].milliseconds, tflops);
        }
        ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        HIP_CHECK(hipFree(allocation_c));
        HIP_CHECK(hipFree(allocation_b));
        HIP_CHECK(hipFree(allocation_a));
        return timings.empty() ? 1 : 0;
    }

    for (int i = 0; i < warmup; ++i) {
        ROCBLAS_CHECK(gemm());
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        ROCBLAS_CHECK(gemm());
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    const double time_ms = elapsed_ms / iterations;
    const double operations = 2.0 * size * size * static_cast<double>(size);
    const double tflops = operations / (time_ms * 1.0e-3) / 1.0e12;
    std::printf("rocBLAS FP16xFP16->%s: %.6f ms, %.3f TFLOPS\n",
                fp16_output ? "FP16" : "FP32", time_ms, tflops);

    // Validate after timing so the copy and host reference cannot affect the
    // performance interval. A deterministic sample is sufficient here: the
    // probe is primarily used to test custom Tensile library overrides, where
    // a bad problem type or leading-dimension mapping otherwise produces a
    // plausible timing with completely wrong output.
    std::vector<float> host_c_f32;
    std::vector<__half> host_c_f16;
    if (fp16_output) {
        host_c_f16.resize(elements);
        HIP_CHECK(hipMemcpy(host_c_f16.data(), device_c, output_bytes,
                            hipMemcpyDeviceToHost));
    } else {
        host_c_f32.resize(elements);
        HIP_CHECK(hipMemcpy(host_c_f32.data(), device_c, output_bytes,
                            hipMemcpyDeviceToHost));
    }

    float max_normalized_error = 0.0f;
    bool all_finite = true;
    constexpr int sample_count = 64;
    for (int sample = 0; sample < sample_count; ++sample) {
        const int row = sample == 0 ? 0
            : sample == 1 ? size - 1
            : static_cast<int>((static_cast<unsigned>(sample) * 104729u)
                               % static_cast<unsigned>(size));
        const int column = sample == 0 ? 0
            : sample == 1 ? size - 1
            : static_cast<int>((static_cast<unsigned>(sample) * 130363u + 17u)
                               % static_cast<unsigned>(size));
        float expected = 0.0f;
        for (int k = 0; k < size; ++k) {
            expected += __half2float(
                            host_a[static_cast<size_t>(row)
                                   * leading_dimension + k])
                * __half2float(host_b[static_cast<size_t>(k)
                                      * leading_dimension + column]);
        }
        const size_t index = static_cast<size_t>(row) * leading_dimension
            + column;
        const float actual = fp16_output
            ? __half2float(host_c_f16[index]) : host_c_f32[index];
        all_finite = all_finite && std::isfinite(actual);
        const float normalized_error = std::fabs(actual - expected)
            / std::max(std::fabs(expected), 1.0f);
        max_normalized_error = std::max(max_normalized_error,
                                        normalized_error);
    }
    std::printf("validation: finite=%s, samples=%d, max normalized error=%.8g\n",
                all_finite ? "yes" : "no", sample_count,
                max_normalized_error);
    const bool validation_failed =
        !all_finite || max_normalized_error >= 0.01f;
    if (validation_failed) {
        std::fprintf(stderr, "validation failed\n");
    }

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(allocation_c));
    HIP_CHECK(hipFree(allocation_b));
    HIP_CHECK(hipFree(allocation_a));
    return validation_failed ? 1 : 0;
}
