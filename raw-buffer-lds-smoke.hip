#include <hip/hip_runtime.h>
#include <stdint.h>

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
using as3_u32_ptr = uint32_t __attribute__((address_space(3)))*;

extern "C" __device__ void raw_buffer_load_lds(
    i32x4 rsrc,
    as3_u32_ptr lds_ptr,
    int size,
    int voffset,
    int soffset,
    int offset,
    int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

__global__ void smoke(uint32_t* output, const uint32_t* input)
{
    __shared__ uint32_t scratch[256];
    const auto resource = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<uint32_t*>(input), 0, 1024, 0x31004000);
    const i32x4 rsrc = *reinterpret_cast<const i32x4*>(&resource);
    const uintptr_t address = reinterpret_cast<uintptr_t>(&scratch[threadIdx.x]);
    raw_buffer_load_lds(rsrc,
                        reinterpret_cast<as3_u32_ptr>(address),
                        4,
                        static_cast<int>(threadIdx.x * 4),
                        0,
                        0,
                        0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    output[threadIdx.x] = scratch[threadIdx.x];
}
