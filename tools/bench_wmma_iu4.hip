// gfx1151 V_WMMA_I32_16X16X16_IU4 issue-rate qualification.
//
// AMD's RDNA3.5 ISA and Matrix Instruction Calculator (commit 2ef9189)
// specify 8192 integer operations in 16 execution cycles, two operand VGPRs
// apiece, and eight INT32 accumulator/result VGPRs.  Lanes 16-31 must replicate
// lanes 0-15.  This benchmark uses packed +1 nibbles in every lane, so every
// output element increases by exactly 16 per instruction.  It is an integer
// TOPS microbenchmark, not an FP16 GEMM result.
//
// Build the default eight-chain form with hipcc -O3 --offload-arch=gfx1151
// -mcumode -std=c++20 tools/bench_wmma_iu4.hip -o bench-wmma-iu4.
// Add -DIU4_CHAINS=16 for the measured peak configuration.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        const hipError_t status = (expr);                                      \
        if(status != hipSuccess)                                               \
        {                                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                 \
                         hipGetErrorString(status));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while(0)

namespace
{

using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

constexpr int kThreads = 256;
constexpr int kWavesPerBlock = kThreads / 32;
#ifndef IU4_CHAINS
#define IU4_CHAINS 8
#endif
constexpr int kChains = IU4_CHAINS;
constexpr int kOpsPerWmma = 2 * 16 * 16 * 16;
static_assert(kChains > 0, "IU4_CHAINS must be positive");

__host__ __device__ constexpr int validation_a(int row, int k)
{
    return ((row * 3 + k * 5 + 1) & 15) - 8;
}

__host__ __device__ constexpr int validation_b(int k, int column)
{
    return ((k * 7 + column * 2 + 3) & 15) - 8;
}

__global__ __launch_bounds__(32) void validate_iu4_layout(int32_t* output)
{
    const int lane = static_cast<int>(threadIdx.x);
    const int replicated_index = lane & 15;
    i32x2 a{};
    i32x2 b{};

    #pragma unroll
    for(int k = 0; k < 16; ++k)
    {
        const int word = k >> 3;
        const int shift = (k & 7) * 4;
        const uint32_t a_nibble
            = static_cast<uint32_t>(validation_a(replicated_index, k)) & 15u;
        const uint32_t b_nibble
            = static_cast<uint32_t>(validation_b(k, replicated_index)) & 15u;
        a[word] |= static_cast<int32_t>(a_nibble << shift);
        b[word] |= static_cast<int32_t>(b_nibble << shift);
    }

    const i32x8 result
        = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
            true, a, true, b, i32x8{}, false);
    #pragma unroll
    for(int element = 0; element < 8; ++element)
        output[lane * 8 + element] = result[element];
}

__global__ __launch_bounds__(kThreads) void bench_iu4(int32_t* output,
                                                      int iterations)
{
    // 0x1 in every nibble encodes sixteen signed +1 values in the two source
    // VGPRs.  Using the same value in every lane satisfies half-wave
    // replication and gives every output element the closed-form result 16*N.
    const i32x2 a = {0x11111111, 0x11111111};
    const i32x2 b = {0x11111111, 0x11111111};
    i32x8 c[kChains] = {};

    #pragma unroll 1
    for(int iteration = 0; iteration < iterations; ++iteration)
    {
        #pragma unroll
        for(int chain = 0; chain < kChains; ++chain)
        {
            c[chain] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                true, a, true, b, c[chain], false);
        }
    }

    int32_t checksum = 0;
    #pragma unroll
    for(int chain = 0; chain < kChains; ++chain)
        checksum += c[chain][0];
    output[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x]
        = checksum;
}

double median(std::vector<float> samples)
{
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if((samples.size() & 1u) != 0u)
        return samples[middle];
    return (static_cast<double>(samples[middle - 1]) + samples[middle]) / 2.0;
}

} // namespace

int main(int argc, char** argv)
{
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 80;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 4096;
    const int warmup = argc > 3 ? std::atoi(argv[3]) : 10;
    const int samples = argc > 4 ? std::atoi(argv[4]) : 20;
    if(blocks < 1 || iterations < 1 || warmup < 1 || samples < 1 || argc > 5)
    {
        std::fprintf(stderr,
                     "usage: %s [blocks] [iterations] [warmup] [samples]\n",
                     argv[0]);
        return 2;
    }

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));

    int32_t* output = nullptr;
    const size_t output_elements = static_cast<size_t>(blocks) * kThreads;
    HIP_CHECK(hipMalloc(&output, output_elements * sizeof(int32_t)));

    // The issue-rate loop's all-ones result checks arithmetic but is invariant
    // to lane placement.  This nonuniform tile separately verifies the ISA's
    // replicated A/B halves and D[even row]/D[odd row] lane mapping.
    validate_iu4_layout<<<1, 32>>>(output);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<int32_t> layout_output(32 * 8);
    HIP_CHECK(hipMemcpy(layout_output.data(),
                        output,
                        layout_output.size() * sizeof(int32_t),
                        hipMemcpyDeviceToHost));
    size_t layout_mismatches = 0;
    for(int lane = 0; lane < 32; ++lane)
    {
        const int column = lane & 15;
        for(int element = 0; element < 8; ++element)
        {
            const int row = element * 2 + (lane >> 4);
            int32_t expected_value = 0;
            for(int k = 0; k < 16; ++k)
                expected_value += validation_a(row, k)
                    * validation_b(k, column);
            layout_mismatches += layout_output[lane * 8 + element]
                    != expected_value
                ? 1u
                : 0u;
        }
    }

    for(int pass = 0; pass < warmup; ++pass)
        bench_iu4<<<blocks, kThreads>>>(output, iterations);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start{};
    hipEvent_t stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    std::vector<float> timings;
    timings.reserve(static_cast<size_t>(samples));
    for(int sample = 0; sample < samples; ++sample)
    {
        HIP_CHECK(hipEventRecord(start));
        bench_iu4<<<blocks, kThreads>>>(output, iterations);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);
    }

    std::vector<int32_t> host_output(output_elements);
    HIP_CHECK(hipMemcpy(host_output.data(),
                        output,
                        output_elements * sizeof(int32_t),
                        hipMemcpyDeviceToHost));
    const int32_t expected = 16 * iterations * kChains;
    size_t mismatches = 0;
    for(const int32_t value : host_output)
        mismatches += value != expected ? 1u : 0u;

    const double median_ms = median(timings);
    const double instructions = static_cast<double>(blocks)
        * kWavesPerBlock * iterations * kChains;
    const double operations = instructions * kOpsPerWmma;
    const double tops = operations / (median_ms * 1.0e9);
    const auto [minimum, maximum]
        = std::minmax_element(timings.begin(), timings.end());

    std::printf("device=%s blocks=%d waves_per_block=%d chains=%d "
                "iterations=%d warmup=%d samples=%d\n",
                properties.name,
                blocks,
                kWavesPerBlock,
                kChains,
                iterations,
                warmup,
                samples);
    std::printf("median_ms=%.6f min_ms=%.6f max_ms=%.6f tops=%.3f\n",
                median_ms,
                static_cast<double>(*minimum),
                static_cast<double>(*maximum),
                tops);
    std::printf("validation expected=%d mismatches=%zu elements=%zu pass=%s\n",
                expected,
                mismatches,
                output_elements,
                mismatches == 0 ? "yes" : "no");
    std::printf("layout_validation mismatches=%zu elements=%zu pass=%s\n",
                layout_mismatches,
                layout_output.size(),
                layout_mismatches == 0 ? "yes" : "no");

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(output));
    return mismatches == 0 && layout_mismatches == 0 ? 0 : 1;
}
