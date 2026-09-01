// Tiled row-major FP16 transpose used by the fair prepacked-B experiment.
// The transpose launch and its fresh output allocation remain inside every
// timed binding call; this is not an amortized or cached-weight benchmark.

#ifndef TRANSPOSE_FP16_TILED_HPP
#define TRANSPOSE_FP16_TILED_HPP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

constexpr int FP16_TRANSPOSE_TILE = 32;
constexpr int FP16_TRANSPOSE_ROWS_PER_THREAD = 4;
constexpr int FP16_TRANSPOSE_THREADS_Y =
    FP16_TRANSPOSE_TILE / FP16_TRANSPOSE_ROWS_PER_THREAD;

__launch_bounds__(FP16_TRANSPOSE_TILE * FP16_TRANSPOSE_THREADS_Y)
__global__ void transpose_fp16_tiled_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int columns
) {
    __shared__ __half tile[FP16_TRANSPOSE_TILE][FP16_TRANSPOSE_TILE + 1];
    const int input_column = blockIdx.x * FP16_TRANSPOSE_TILE + threadIdx.x;
    const int input_row_base =
        blockIdx.y * FP16_TRANSPOSE_TILE + threadIdx.y;

    #pragma unroll
    for (int offset = 0; offset < FP16_TRANSPOSE_TILE;
         offset += FP16_TRANSPOSE_THREADS_Y) {
        const int input_row = input_row_base + offset;
        if (input_row < rows && input_column < columns) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[input_row * columns + input_column];
        }
    }
    __syncthreads();

    const int output_column =
        blockIdx.y * FP16_TRANSPOSE_TILE + threadIdx.x;
    const int output_row_base =
        blockIdx.x * FP16_TRANSPOSE_TILE + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < FP16_TRANSPOSE_TILE;
         offset += FP16_TRANSPOSE_THREADS_Y) {
        const int output_row = output_row_base + offset;
        if (output_row < columns && output_column < rows) {
            output[output_row * rows + output_column] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

#endif
