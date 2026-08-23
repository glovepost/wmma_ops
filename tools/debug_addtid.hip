#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t status = (expr); \
    if (status != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(status)); \
        std::exit(1); \
    } \
} while (0)

using u16x8 = uint16_t __attribute__((ext_vector_type(8)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));

__global__ void debug_addtid(const uint16_t* input, uint16_t* output)
{
    constexpr unsigned config = 0x31004000u | (1u << 23) | (1u << 21);
    __amdgpu_buffer_rsrc_t resource = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<uint16_t*>(input), 16, 256, config);
    u32x4 descriptor;
    __builtin_memcpy(&descriptor, &resource, sizeof(descriptor));
    descriptor[1] |= 3u << 30;
    __builtin_memcpy(&resource, &descriptor, sizeof(descriptor));

    using payload = float __attribute__((ext_vector_type(4)));
    payload raw = __builtin_amdgcn_raw_buffer_load_b128(resource, 0, 0, 0);
    u16x8 values;
    __builtin_memcpy(&values, &raw, sizeof(values));
    const int lane = static_cast<int>(threadIdx.x);
    #pragma unroll
    for(int item = 0; item < 8; ++item)
        output[lane * 8 + item] = values[item];
}

int main()
{
    constexpr int rows = 16;
    constexpr int columns = 128;
    std::vector<uint16_t> input(rows * columns);
    std::vector<uint16_t> output(32 * 8);
    for(int row = 0; row < rows; ++row)
        for(int column = 0; column < columns; ++column)
            input[row * columns + column]
                = static_cast<uint16_t>(row * columns + column);

    uint16_t* device_input = nullptr;
    uint16_t* device_output = nullptr;
    HIP_CHECK(hipMalloc(&device_input, input.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&device_output, output.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(device_input, input.data(), input.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice));
    hipLaunchKernelGGL(debug_addtid, dim3(1), dim3(32), 0, 0,
                       device_input, device_output);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(output.data(), device_output,
                        output.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));

    int mismatches = 0;
    for(int lane = 0; lane < 32; ++lane)
    {
        const int expected_row = lane / 16;
        const int expected_column = (lane % 16) * 8;
        std::printf("lane=%d got=", lane);
        for(int item = 0; item < 8; ++item)
            std::printf("%s%u", item ? "," : "",
                        static_cast<unsigned>(output[lane * 8 + item]));
        std::printf(" expected=%d..%d\n",
                    expected_row * columns + expected_column,
                    expected_row * columns + expected_column + 7);
        for(int item = 0; item < 8; ++item)
            mismatches += output[lane * 8 + item]
                          != expected_row * columns + expected_column + item;
    }
    std::printf("mismatches=%d\n", mismatches);
    HIP_CHECK(hipFree(device_output));
    HIP_CHECK(hipFree(device_input));
    return mismatches == 0 ? 0 : 1;
}
