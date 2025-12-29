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
 * @brief L2/LLC-aware Chiplet Swizzling for AMD Multi-XCD GPUs
 * 
 * Based on HipKittens (arXiv:2511.08083) Algorithm 1.
 * Optimizes cache reuse on AMD GPUs with multiple XCDs (chiplets).
 * 
 * Key concepts:
 * 1. XCD Grouping: Consecutive block IDs are mapped to the same XCD
 *    to reduce cross-chiplet traffic.
 * 2. Hierarchical Windowed Traversal: Process grid in vertical windows
 *    to maximize L2 cache reuse.
 * 
 * Parameters:
 * - W (window_height): Height of L2 tile window (e.g., 4 or 8)
 * - C (chunk_size): Number of consecutive blocks per XCD
 * - num_xcds: Number of XCDs on the GPU (e.g., 8 for MI355X)
 * 
 * The algorithm remaps block IDs to optimize:
 * - L2 reuse: Blocks on same XCD work on adjacent output tiles
 * - LLC reuse: Multiple XCDs work on overlapping input regions
 */
template<int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
chiplet_swizzle_tile_mapping(
    int tile_id,
    int grid_m,
    int grid_n,
    int window_height,  // W: L2 tile height (e.g., 4 or 8)
    int chunk_size,     // C: blocks per XCD chunk
    int num_xcds,       // Number of XCDs (8 for MI355X, 1 for RDNA)
    int* block_row,
    int* block_col)
{
    int total_tiles = grid_m * grid_n;
    if (tile_id >= total_tiles) {
        *block_row = 0;
        *block_col = 0;
        return;
    }
    
    // For single-XCD GPUs (RDNA3/3.5), use simpler windowed traversal
    if (num_xcds <= 1) {
        // Simple windowed traversal without XCD grouping
        int window_id = tile_id / (window_height * grid_n);
        int local_id = tile_id % (window_height * grid_n);
        
        int row_in_window = local_id % window_height;
        int col = local_id / window_height;
        
        int row = window_id * window_height + row_in_window;
        
        // Clamp to grid bounds
        row = min(row, grid_m - 1);
        col = min(col, grid_n - 1);
        
        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
        return;
    }
    
    // ========================================================================
    // Multi-XCD Chiplet Swizzling (Algorithm 1 from HipKittens)
    // ========================================================================
    
    // Step 1: XCD Grouping
    // Remap block IDs so chunks of C consecutive IDs go to same XCD
    int xcd_id = (tile_id / chunk_size) % num_xcds;
    int chunk_offset = tile_id % chunk_size;
    int chunk_group = tile_id / (chunk_size * num_xcds);
    
    // Compute the effective block ID within this XCD's work
    int blocks_per_xcd = (total_tiles + num_xcds - 1) / num_xcds;
    int xcd_local_id = chunk_group * chunk_size + chunk_offset;
    
    if (xcd_local_id >= blocks_per_xcd) {
        // Handle remainder blocks
        xcd_local_id = xcd_local_id % blocks_per_xcd;
    }
    
    // Step 2: Hierarchical Windowed Traversal
    // Process in vertical windows of height W for L2 reuse
    int tiles_per_window = window_height * grid_n;
    int window_id = xcd_local_id / tiles_per_window;
    int local_in_window = xcd_local_id % tiles_per_window;
    
    // Within a window, traverse column-major for better B matrix reuse
    int row_in_window = local_in_window % window_height;
    int col = local_in_window / window_height;
    
    int row = window_id * window_height + row_in_window;
    
    // Clamp to grid bounds
    row = min(row, grid_m - 1);
    col = min(col, grid_n - 1);
    
    *block_row = row * BLOCK_M;
    *block_col = col * BLOCK_N;
}

/**
 * @brief Simple 2D swizzle for L2 locality without XCD awareness
 * 
 * Tiles are grouped into 2D "super-tiles" that are processed together.
 * This provides L2 locality benefits on single-XCD GPUs like RDNA3.
 */
template<int L2_TILE_M, int L2_TILE_N, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
l2_aware_tile_mapping(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
{
    int total_tiles = grid_m * grid_n;
    if (tile_id >= total_tiles) {
        *block_row = 0;
        *block_col = 0;
        return;
    }
    
    // Number of super-tiles in each dimension
    int super_grid_m = (grid_m + L2_TILE_M - 1) / L2_TILE_M;
    int super_grid_n = (grid_n + L2_TILE_N - 1) / L2_TILE_N;
    
    // Size of each super-tile (may be smaller at boundaries)
    int tiles_per_super = L2_TILE_M * L2_TILE_N;
    
    // Which super-tile and position within it
    int super_id = tile_id / tiles_per_super;
    int local_id = tile_id % tiles_per_super;
    
    // Super-tile coordinates (column-major ordering of super-tiles)
    int super_row = super_id % super_grid_m;
    int super_col = super_id / super_grid_m;
    
    // Local coordinates within super-tile (column-major for B reuse)
    int local_row = local_id % L2_TILE_M;
    int local_col = local_id / L2_TILE_M;
    
    // Global tile coordinates
    int row = super_row * L2_TILE_M + local_row;
    int col = super_col * L2_TILE_N + local_col;
    
    // Clamp to valid range
    row = min(row, grid_m - 1);
    col = min(col, grid_n - 1);
    
    *block_row = row * BLOCK_M;
    *block_col = col * BLOCK_N;
}

/**
 * @brief Select optimal tile mapping strategy based on matrix shape
 */
enum class TileMappingMode {
    ROW_MAJOR,       // Standard row-major (baseline)
    HILBERT,         // Hilbert curve for square matrices
    SWIZZLE,         // Simple swizzle for rectangular matrices
    CHIPLET_SWIZZLE, // L2/LLC-aware chiplet swizzling (multi-XCD)
    L2_AWARE         // L2-aware 2D super-tile mapping
};

/**
 * @brief Get recommended chiplet swizzle parameters for target GPU
 * 
 * @param gpu_arch GPU architecture string (e.g., "gfx1151", "gfx942")
 * @param[out] num_xcds Number of XCDs
 * @param[out] window_height Recommended L2 window height
 * @param[out] chunk_size Recommended XCD chunk size
 */
__host__ inline void get_chiplet_params(
    const char* gpu_arch,
    int* num_xcds,
    int* window_height,
    int* chunk_size)
{
    // Default for single-XCD GPUs (RDNA3/3.5)
    *num_xcds = 1;
    *window_height = 4;
    *chunk_size = 32;
    
    // MI300X / MI325X (CDNA3): 8 XCDs, 38 CUs per XCD
    if (strstr(gpu_arch, "gfx942") != nullptr) {
        *num_xcds = 8;
        *window_height = 4;  // 4x8 or 8x4 L2 tiles work well
        *chunk_size = 32;    // ~32 blocks per XCD chunk
    }
    
    // MI350X / MI355X (CDNA4): 8 XCDs, 32 CUs per XCD
    if (strstr(gpu_arch, "gfx950") != nullptr) {
        *num_xcds = 8;
        *window_height = 8;
        *chunk_size = 32;
    }
    
    // gfx1151 (RDNA3.5 Strix Halo): Single die, no XCDs
    // L2-aware mapping is still beneficial
    if (strstr(gpu_arch, "gfx1151") != nullptr) {
        *num_xcds = 1;
        *window_height = 4;
        *chunk_size = 16;
    }
}

__host__ inline TileMappingMode select_tile_mapping(int M, int N, int BLOCK_M, int BLOCK_N)
{
    int grid_m = (M + BLOCK_M - 1) / BLOCK_M;
    int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
    
    // For small matrices, row-major is fine
    if (grid_m * grid_n < 64) {
        return TileMappingMode::ROW_MAJOR;
    }
    
    // For large matrices, prefer L2-aware mapping
    if (grid_m * grid_n >= 256) {
        return TileMappingMode::L2_AWARE;
    }
    
    // For roughly square matrices, Hilbert is good
    float aspect = static_cast<float>(grid_m) / static_cast<float>(grid_n);
    if (aspect >= 0.5f && aspect <= 2.0f && grid_m >= 8 && grid_n >= 8) {
        return TileMappingMode::HILBERT;
    }
    
    // For rectangular matrices, swizzle provides some benefit
    return TileMappingMode::SWIZZLE;
}

} // namespace tile_mapping

#endif // WMMA_TILE_MAPPING_HPP
