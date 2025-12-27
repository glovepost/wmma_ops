// ============================================================================
// WMMA TILE MAPPING UTILITIES
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
//
// Provides optimized tile mapping strategies for better L2 cache locality:
// - Hilbert curve mapping for large square matrices
// - Swizzle-based mapping for rectangular matrices
//
// Based on: adelj88/rocm_wmma_samples/hgemm/include/kernels/common.hpp
// ============================================================================

#ifndef WMMA_TILE_MAPPING_HPP
#define WMMA_TILE_MAPPING_HPP

#include <hip/hip_runtime.h>
#include <cstdint>

namespace tile_mapping {

// ============================================================================
// HILBERT CURVE UTILITIES
// ============================================================================

/**
 * @brief Optimized Hilbert curve d2xy mapping using bit manipulation
 *
 * Converts a distance along the Hilbert curve to (x,y) coordinates using
 * GPU-optimized bit manipulation techniques.
 *
 * @param[in]  n      Size of the grid (must be a power of 2)
 * @param[in]  index  Distance along the Hilbert curve
 * @param[out] x      Resulting x coordinate
 * @param[out] y      Resulting y coordinate
 */
__device__ __forceinline__ void
hilbert_d2xy_optimized(uint32_t n, uint32_t index, uint32_t* x, uint32_t* y)
{
    *x = 0;
    *y = 0;

    #pragma unroll 16  // Unroll for better instruction-level parallelism
    for (uint32_t i = 0; i < 16; ++i) {  // Max 2^16 x 2^16 grid
        if ((n >> i) == 0) {
            break;  // Early termination
        }

        // Extract 2 bits from index
        uint32_t bits = (index >> (i * 2)) & 3;

        // Use lookup table approach for the rotation logic (better for GPU)
        switch (bits) {
            case 0: {  // Lower left quadrant (reflect and swap)
                uint32_t temp = *x;
                *x = *y;
                *y = temp;
                break;
            }
            case 1: {  // Lower right quadrant
                *y |= (1U << i);
                break;
            }
            case 2: {  // Upper right quadrant
                *x |= (1U << i);
                *y |= (1U << i);
                break;
            }
            case 3: {  // Upper left quadrant (reflect and swap)
                uint32_t temp = (1U << i) - 1 - *y;
                *y = (1U << i) - 1 - *x;
                *x = temp;
                *x |= (1U << i);
                break;
            }
        }
    }
}

/**
 * @brief Find the largest power of 2 not exceeding n
 * Uses efficient bit manipulation instead of looping.
 */
__device__ __forceinline__ uint32_t largest_power_of_2(uint32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

// ============================================================================
// TILE MAPPING FUNCTIONS
// ============================================================================

/**
 * @brief Hilbert curve tile mapping with core + remainder approach
 *
 * Computes block indices using a hybrid approach:
 * 1. A power-of-two core grid uses optimal Hilbert curve for spatial locality
 * 2. Remainder regions use simpler linear mapping
 *
 * @param[in]  tile_id    Linear block ID
 * @param[in]  grid_m     Number of blocks in M dimension
 * @param[in]  grid_n     Number of blocks in N dimension  
 * @param[in]  BLOCK_M    Block size in M dimension
 * @param[in]  BLOCK_N    Block size in N dimension
 * @param[out] block_row  Computed block row coordinate (in elements, not tiles)
 * @param[out] block_col  Computed block column coordinate (in elements, not tiles)
 */
template<int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
hilbert_tile_mapping(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
{
    // Check bounds
    int total_tiles = grid_m * grid_n;
    if (tile_id >= total_tiles) {
        *block_row = 0;
        *block_col = 0;
        return;
    }

    // Special fast path for perfect power-of-two square grids
    if (grid_m == grid_n && (grid_m & (grid_m - 1)) == 0) {
        // Direct optimized Hilbert calculation - no remainder needed
        uint32_t x, y;
        hilbert_d2xy_optimized(grid_m, tile_id, &x, &y);

        // Convert to actual block coordinates with bit shift for multiplication
        *block_row = y << (__ffs(BLOCK_M) - 1);
        *block_col = x << (__ffs(BLOCK_N) - 1);
        return;
    }

    // Find the largest power-of-two dimensions that fit within our grid
    uint32_t core_m = largest_power_of_2(grid_m);
    uint32_t core_n = largest_power_of_2(grid_n);

    // Calculate size of the core grid (power-of-two in both dimensions)
    uint32_t core_size = min(core_m, core_n);
    uint32_t core_tiles = core_size * core_size;

    // Check if we're in the core grid or in the remainder
    if (tile_id < (int)core_tiles) {
        // We're in the power-of-two core grid - use optimized Hilbert curve
        uint32_t x, y;
        hilbert_d2xy_optimized(core_size, tile_id, &x, &y);

        // Convert to actual block coordinates
        *block_row = y * BLOCK_M;
        *block_col = x * BLOCK_N;
    } else {
        // We're in the remainder regions - use simpler mapping
        int remainder_id = tile_id - core_tiles;

        // Define the three remainder regions
        int right_region_width = grid_n - core_size;
        int bottom_region_height = grid_m - core_size;
        int right_region_tiles = core_size * right_region_width;
        int bottom_region_tiles = bottom_region_height * core_size;

        int row, col;

        // Map to the appropriate region
        if (remainder_id < right_region_tiles) {
            // Right region - core_size rows, right_region_width columns
            row = remainder_id / right_region_width;
            col = core_size + (remainder_id % right_region_width);
        } else if (remainder_id < right_region_tiles + bottom_region_tiles) {
            // Bottom region - bottom_region_height rows, core_size columns
            int local_id = remainder_id - right_region_tiles;
            row = core_size + (local_id / core_size);
            col = local_id % core_size;
        } else {
            // Corner region - bottom_region_height rows, right_region_width columns
            int local_id = remainder_id - right_region_tiles - bottom_region_tiles;
            row = core_size + (local_id / right_region_width);
            col = core_size + (local_id % right_region_width);
        }

        // Convert to actual block coordinates
        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
    }
}

/**
 * @brief Swizzle-based tile mapping for better L2 locality
 * 
 * Groups tiles along the M dimension for better cache reuse.
 * Simpler than Hilbert but still provides locality benefits.
 */
template<int GROUP_SIZE, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
swizzle_tile_mapping(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
{
    // Group tiles along the M dimension for better cache locality
    int width = GROUP_SIZE * grid_n;
    int group_id = tile_id / width;
    int group_size = min(GROUP_SIZE, grid_m - group_id * GROUP_SIZE);

    // Compute swizzled indices
    int pid_m = group_id * GROUP_SIZE + (tile_id % group_size);
    int pid_n = (tile_id % width) / group_size;

    // Convert to actual block coordinates
    *block_row = pid_m * BLOCK_M;
    *block_col = pid_n * BLOCK_N;
}

/**
 * @brief Select optimal tile mapping strategy based on matrix shape
 */
enum class TileMappingMode {
    ROW_MAJOR,   // Standard row-major (baseline)
    HILBERT,     // Hilbert curve for square matrices
    SWIZZLE      // Swizzle for rectangular matrices
};

__host__ inline TileMappingMode select_tile_mapping(int M, int N, int BLOCK_M, int BLOCK_N)
{
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    
    // For small matrices, row-major is fine
    if (grid_m * grid_n < 64) {
        return TileMappingMode::ROW_MAJOR;
    }
    
    // For roughly square matrices, Hilbert is best
    float aspect = static_cast<float>(grid_m) / static_cast<float>(grid_n);
    if (aspect >= 0.5f && aspect <= 2.0f && grid_m >= 8 && grid_n >= 8) {
        return TileMappingMode::HILBERT;
    }
    
    // For rectangular matrices, swizzle provides some benefit
    return TileMappingMode::SWIZZLE;
}

} // namespace tile_mapping

#endif // WMMA_TILE_MAPPING_HPP
