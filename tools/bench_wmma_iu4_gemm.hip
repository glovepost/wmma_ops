// Prepacked linear W4A4 GEMM for gfx1151.
//
// This is a new integer numerical contract, not a reinterpretation of
// ROCmFP4 Codebook10.  A and B contain signed linear INT4 values packed in
// block/K-major native-WMMA order.  The kernel computes exact INT32 output.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        const hipError_t status = (expr);                                      \
        if(status != hipSuccess)                                               \
        {                                                                      \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                   \
                         hipGetErrorString(status));                           \
            std::exit(1);                                                      \
        }                                                                      \
    } while(0)

namespace
{

using i32x2 = int32_t __attribute__((ext_vector_type(2)));
using i32x8 = int32_t __attribute__((ext_vector_type(8)));

#ifndef IU4_BLOCK_M
#define IU4_BLOCK_M 128
#endif
constexpr int kBlockM = IU4_BLOCK_M;
constexpr int kBlockN = 128;
constexpr int kBlockK = 16;
constexpr int kThreads = 256;
#ifndef IU4_LDS_STRIDE_WORDS
#define IU4_LDS_STRIDE_WORDS 2
#endif
constexpr int kLdsStrideWords = IU4_LDS_STRIDE_WORDS;
static_assert(kLdsStrideWords >= 2);
static_assert(kBlockM == 128 || kBlockM == 256);
#ifndef IU4_WAVE_COLUMNS
#define IU4_WAVE_COLUMNS 4
#endif
#ifndef IU4_PAIR_K
#define IU4_PAIR_K 0
#endif
constexpr int kWaveColumns = IU4_WAVE_COLUMNS;
static_assert(kWaveColumns == 2 || kWaveColumns == 4);
constexpr int kWaveRows = (kThreads / 32) / kWaveColumns;
constexpr int kWaveTilesM = kBlockM / (kWaveRows * kBlockK);
constexpr int kWaveTilesN = kBlockN / (kWaveColumns * kBlockK);
constexpr int kCooperativeAWords = kBlockM * 2 / kThreads;

__host__ __device__ constexpr int value_a(int row, int k)
{
    return ((row * 3 + k * 5 + 1) & 15) - 8;
}

__host__ __device__ constexpr int value_b(int k, int column)
{
    return ((k * 7 + column * 2 + 3) & 15) - 8;
}

__global__ __launch_bounds__(kThreads) void iu4_gemm(
    int32_t* __restrict__ c,
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    int n,
    int k)
{
    constexpr int kPairSlices = IU4_PAIR_K > 0 ? IU4_PAIR_K * 2 : 2;
    constexpr int kLdsBuffers = IU4_PAIR_K ? kPairSlices * 2 : 2;
    static_assert(IU4_PAIR_K == 0 || IU4_PAIR_K == 1 || IU4_PAIR_K == 2
                  || IU4_PAIR_K == 4);
    __shared__ uint32_t a_lds[kLdsBuffers][kBlockM][kLdsStrideWords];
    __shared__ uint32_t b_lds[kLdsBuffers][kBlockN][kLdsStrideWords];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int wave = tid >> 5;
    const int wave_row = wave / kWaveColumns;
    const int wave_column = wave % kWaveColumns;
    const int grid_n = n / kBlockN;
    const int block_m = static_cast<int>(blockIdx.x) / grid_n;
    const int block_n = static_cast<int>(blockIdx.x) - block_m * grid_n;
    const int k_tiles = k / kBlockK;
    const size_t a_block = static_cast<size_t>(block_m) * k_tiles
        * kBlockM * 2;
    const size_t b_block = static_cast<size_t>(block_n) * k_tiles
        * kBlockN * 2;

    #pragma unroll
    for(int load = 0; load < kCooperativeAWords; ++load)
    {
        const int flat = tid * kCooperativeAWords + load;
        a_lds[0][flat >> 1][flat & 1]
            = a[a_block + static_cast<size_t>(flat)];
    }
    const int cooperative_row = tid >> 1;
    const int cooperative_word = tid & 1;
    b_lds[0][cooperative_row][cooperative_word]
        = b[b_block + static_cast<size_t>(tid)];
    if constexpr(IU4_PAIR_K)
    {
        #pragma unroll
        for(int init_tile = 1; init_tile < kPairSlices; ++init_tile)
        {
            #pragma unroll
            for(int load = 0; load < kCooperativeAWords; ++load)
            {
                const int flat = tid * kCooperativeAWords + load;
                a_lds[init_tile][flat >> 1][flat & 1]
                    = a[a_block + static_cast<size_t>(init_tile * kBlockM * 2
                                                       + flat)];
            }
            b_lds[init_tile][cooperative_row][cooperative_word]
                = b[b_block + static_cast<size_t>(init_tile * kBlockN * 2
                                                   + tid)];
        }
    }
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();

    i32x8 accum[kWaveTilesM][kWaveTilesN] = {};

    if constexpr(IU4_PAIR_K)
    {
        // Keep two complete K16 slices resident while the next pair is
        // fetched into the unused LDS slots.  The only barrier is at the
        // pair boundary, so the two WMMA groups share one publication cost.
        for(int pair_tile = 0; pair_tile < k_tiles;
            pair_tile += kPairSlices)
        {
            const int current = ((pair_tile / kPairSlices) & 1)
                * kPairSlices;
            const int next = current ^ kPairSlices;
            const bool has_next_pair = pair_tile + kPairSlices < k_tiles;
            uint32_t staged_a[kPairSlices][kCooperativeAWords]{};
            uint32_t staged_b[kPairSlices]{};
            if(has_next_pair)
            {
                #pragma unroll
                for(int tile_offset = 0; tile_offset < kPairSlices;
                    ++tile_offset)
                {
                    const size_t tile = static_cast<size_t>(pair_tile
                        + kPairSlices + tile_offset);
                    #pragma unroll
                    for(int load = 0; load < kCooperativeAWords; ++load)
                    {
                        const int flat = tid * kCooperativeAWords + load;
                        staged_a[tile_offset][load]
                            = a[a_block + tile * kBlockM * 2
                                + static_cast<size_t>(flat)];
                    }
                    const size_t b_tile = tile * kBlockN * 2;
                    staged_b[tile_offset]
                        = b[b_block + b_tile + static_cast<size_t>(tid)];
                }
            }

            #pragma unroll
            for(int tile_offset = 0; tile_offset < kPairSlices;
                ++tile_offset)
            {
                const int tile = current + tile_offset;
                i32x2 a_frag[kWaveTilesM];
                i32x2 b_frag[kWaveTilesN];
                #pragma unroll
                for(int wm = 0; wm < kWaveTilesM; ++wm)
                {
                    const int row = wave_row * kWaveTilesM * 16
                        + wm * 16 + lane16;
                    a_frag[wm][0]
                        = static_cast<int32_t>(a_lds[tile][row][0]);
                    a_frag[wm][1]
                        = static_cast<int32_t>(a_lds[tile][row][1]);
                }
                #pragma unroll
                for(int wn = 0; wn < kWaveTilesN; ++wn)
                {
                    const int column = wave_column * kWaveTilesN * 16
                        + wn * 16 + lane16;
                    b_frag[wn][0]
                        = static_cast<int32_t>(b_lds[tile][column][0]);
                    b_frag[wn][1]
                        = static_cast<int32_t>(b_lds[tile][column][1]);
                }
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                #pragma unroll
                for(int wm = 0; wm < kWaveTilesM; ++wm)
                {
                    #pragma unroll
                    for(int wn = 0; wn < kWaveTilesN; ++wn)
                    {
                        accum[wm][wn]
                            = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                                true, a_frag[wm], true, b_frag[wn],
                                accum[wm][wn], false);
                    }
                }
            }

            if(has_next_pair)
            {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                #pragma unroll
                for(int tile_offset = 0; tile_offset < kPairSlices;
                    ++tile_offset)
                {
                    #pragma unroll
                    for(int load = 0; load < kCooperativeAWords; ++load)
                    {
                        const int flat = tid * kCooperativeAWords + load;
                        a_lds[next + tile_offset][flat >> 1][flat & 1]
                            = staged_a[tile_offset][load];
                    }
                    b_lds[next + tile_offset][cooperative_row][cooperative_word]
                        = staged_b[tile_offset];
                }
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                __syncthreads();
            }
        }
    }
    else
    {
    #pragma unroll 1
    for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
    {
        const int current = k_tile & 1;
        const int next = current ^ 1;
        const bool has_next = k_tile + 1 < k_tiles;

        i32x2 a_frag[kWaveTilesM];
        i32x2 b_frag[kWaveTilesN];
        #pragma unroll
        for(int wm = 0; wm < kWaveTilesM; ++wm)
        {
            const int row = wave_row * kWaveTilesM * 16 + wm * 16 + lane16;
            a_frag[wm][0] = static_cast<int32_t>(a_lds[current][row][0]);
            a_frag[wm][1] = static_cast<int32_t>(a_lds[current][row][1]);
        }
        #pragma unroll
        for(int wn = 0; wn < kWaveTilesN; ++wn)
        {
            const int column = wave_column * kWaveTilesN * 16
                + wn * 16 + lane16;
            b_frag[wn][0] = static_cast<int32_t>(b_lds[current][column][0]);
            b_frag[wn][1] = static_cast<int32_t>(b_lds[current][column][1]);
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");

        uint32_t staged_a[kCooperativeAWords]{};
        uint32_t staged_b = 0;
        if(has_next)
        {
            const size_t tile_offset = static_cast<size_t>(k_tile + 1)
                * kBlockM * 2;
            #pragma unroll
            for(int load = 0; load < kCooperativeAWords; ++load)
            {
                const int flat = tid * kCooperativeAWords + load;
                staged_a[load]
                    = a[a_block + tile_offset + static_cast<size_t>(flat)];
            }
            const size_t b_tile_offset = static_cast<size_t>(k_tile + 1)
                * kBlockN * 2;
            staged_b = b[b_block + b_tile_offset + static_cast<size_t>(tid)];
        }

        #pragma unroll
        for(int wm = 0; wm < kWaveTilesM; ++wm)
        {
            #pragma unroll
            for(int wn = 0; wn < kWaveTilesN; ++wn)
            {
                accum[wm][wn]
                    = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true,
                        a_frag[wm],
                        true,
                        b_frag[wn],
                        accum[wm][wn],
                        false);
            }
        }

        if(has_next)
        {
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            #pragma unroll
            for(int load = 0; load < kCooperativeAWords; ++load)
            {
                const int flat = tid * kCooperativeAWords + load;
                a_lds[next][flat >> 1][flat & 1] = staged_a[load];
            }
            b_lds[next][cooperative_row][cooperative_word] = staged_b;
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            __syncthreads();
        }
    }
    }

    #pragma unroll
    for(int wm = 0; wm < kWaveTilesM; ++wm)
    {
        #pragma unroll
        for(int wn = 0; wn < kWaveTilesN; ++wn)
        {
            #pragma unroll
            for(int element = 0; element < 8; ++element)
            {
                const int row = block_m * kBlockM
                    + wave_row * kWaveTilesM * 16
                    + wm * 16 + element * 2 + (lane >> 4);
                const int column = block_n * kBlockN
                    + wave_column * kWaveTilesN * 16
                    + wn * 16 + lane16;
                c[static_cast<size_t>(row) * n + column]
                    = accum[wm][wn][element];
            }
        }
    }
}

uint32_t pack_a_word(int global_row, int k_base)
{
    uint32_t packed = 0;
    for(int nibble = 0; nibble < 8; ++nibble)
        packed |= (static_cast<uint32_t>(value_a(global_row, k_base + nibble))
                   & 15u)
            << (4 * nibble);
    return packed;
}

uint32_t pack_b_word(int k_base, int global_column)
{
    uint32_t packed = 0;
    for(int nibble = 0; nibble < 8; ++nibble)
        packed |= (static_cast<uint32_t>(value_b(k_base + nibble,
                                                global_column))
                   & 15u)
            << (4 * nibble);
    return packed;
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
    const int size = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
    const int samples = argc > 3 ? std::atoi(argv[3]) : 20;
    if(size < kBlockM || size % kBlockM != 0 || warmup < 1 || samples < 1
       || argc > 4)
    {
        std::fprintf(stderr, "usage: %s [size_multiple_of_128] [warmup] "
                             "[samples]\n", argv[0]);
        return 2;
    }

    const int k_tiles = size / kBlockK;
    const int blocks_m = size / kBlockM;
    const int blocks_n = size / kBlockN;
    const size_t packed_words = static_cast<size_t>(size) * size / 8;
    std::vector<uint32_t> host_a(packed_words);
    std::vector<uint32_t> host_b(packed_words);
    for(int block = 0; block < blocks_m; ++block)
    {
        for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            for(int row = 0; row < kBlockM; ++row)
            {
                for(int word = 0; word < 2; ++word)
                {
                    const size_t index = (((static_cast<size_t>(block) * k_tiles
                                            + k_tile) * kBlockM + row) * 2)
                        + word;
                    host_a[index] = pack_a_word(block * kBlockM + row,
                                                k_tile * kBlockK + word * 8);
                }
            }
        }
    }
    for(int block = 0; block < blocks_n; ++block)
    {
        for(int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            for(int column = 0; column < kBlockN; ++column)
            {
                for(int word = 0; word < 2; ++word)
                {
                    const size_t index = (((static_cast<size_t>(block) * k_tiles
                                            + k_tile) * kBlockN + column) * 2)
                        + word;
                    host_b[index] = pack_b_word(k_tile * kBlockK + word * 8,
                                                block * kBlockN + column);
                }
            }
        }
    }

    uint32_t* device_a = nullptr;
    uint32_t* device_b = nullptr;
    int32_t* device_c = nullptr;
    const size_t packed_bytes = packed_words * sizeof(uint32_t);
    const size_t output_elements = static_cast<size_t>(size) * size;
    HIP_CHECK(hipMalloc(&device_a, packed_bytes));
    HIP_CHECK(hipMalloc(&device_b, packed_bytes));
    HIP_CHECK(hipMalloc(&device_c, output_elements * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(device_a, host_a.data(), packed_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), packed_bytes,
                        hipMemcpyHostToDevice));

    const int blocks = blocks_m * blocks_n;
    for(int pass = 0; pass < warmup; ++pass)
        iu4_gemm<<<blocks, kThreads>>>(device_c, device_a, device_b,
                                      size, size);
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
        iu4_gemm<<<blocks, kThreads>>>(device_c, device_a, device_b,
                                      size, size);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float milliseconds = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
        timings.push_back(milliseconds);
    }

    std::vector<int32_t> host_c(output_elements);
    HIP_CHECK(hipMemcpy(host_c.data(), device_c,
                        output_elements * sizeof(int32_t),
                        hipMemcpyDeviceToHost));
    int32_t expected[16][16]{};
    for(int row = 0; row < 16; ++row)
    {
        for(int column = 0; column < 16; ++column)
        {
            for(int k = 0; k < 16; ++k)
                expected[row][column] += value_a(row, k) * value_b(k, column);
            expected[row][column] *= size / 16;
        }
    }
    size_t mismatches = 0;
    for(int row = 0; row < size; ++row)
    {
        for(int column = 0; column < size; ++column)
        {
            mismatches += host_c[static_cast<size_t>(row) * size + column]
                    != expected[row & 15][column & 15]
                ? 1u
                : 0u;
        }
    }

    const double median_ms = median(timings);
    const double operations = 2.0 * size * size * size;
    const double tops = operations / (median_ms * 1.0e9);
    const auto [minimum, maximum]
        = std::minmax_element(timings.begin(), timings.end());
    std::printf("shape=%dx%dx%d blocks=%d threads=%d warmup=%d samples=%d\n",
                size, size, size, blocks, kThreads, warmup, samples);
    std::printf("median_ms=%.6f min_ms=%.6f max_ms=%.6f int4_tops=%.3f\n",
                median_ms,
                static_cast<double>(*minimum),
                static_cast<double>(*maximum),
                tops);
    std::printf("validation mismatches=%zu elements=%zu pass=%s\n",
                mismatches,
                output_elements,
                mismatches == 0 ? "yes" : "no");

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipFree(device_c));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_a));
    return mismatches == 0 ? 0 : 1;
}
