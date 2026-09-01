// Counter-collection driver for the record-shape interleaved kernel.
//
// Build this with the same hipcc flags as the extension, then run it only under
// a profiler.  Counter collection serializes dispatches, so its durations are
// never record timings.

#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>

#include "wmma_kernel_opt_interleaved.hpp"

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t error = (call);                                       \
        if (error != hipSuccess) {                                             \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         hipGetErrorString(error));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

int main() {
    constexpr int m = 4096;
    constexpr int n = 4096;
    constexpr int k = 4096;
    constexpr int dispatches = 12;

    __half* a = nullptr;
    __half* b = nullptr;
    float* c = nullptr;
    HIP_CHECK(hipMalloc(&a, static_cast<size_t>(m) * k * sizeof(*a)));
    HIP_CHECK(hipMalloc(&b, static_cast<size_t>(k) * n * sizeof(*b)));
    HIP_CHECK(hipMalloc(&c, static_cast<size_t>(m) * n * sizeof(*c)));
    HIP_CHECK(hipMemset(a, 0, static_cast<size_t>(m) * k * sizeof(*a)));
    HIP_CHECK(hipMemset(b, 0, static_cast<size_t>(k) * n * sizeof(*b)));

    const dim3 grid(n / OptConfig::BLOCK_N, m / OptConfig::BLOCK_M);
    const dim3 block(OptConfig::NUM_THREADS);
    for (int iteration = 0; iteration < dispatches; ++iteration) {
        wmma_gemm_kernel_opt_interleaved<false><<<grid, block>>>(a, b, c, n, k);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(c));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(a));
    return 0;
}
