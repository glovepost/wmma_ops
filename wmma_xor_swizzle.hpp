// ============================================================================
// WMMA GEMM OPTIMIZATIONS - XOR SWIZZLE IMPLEMENTATIONS
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
// 
// Combines XOR swizzle optimizations, L2 cache rasterization, and Split-K
// ============================================================================

#ifndef WMMA_XOR_SWIZZLE_HPP
#define WMMA_XOR_SWIZZLE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>

// ============================================================================
// VECTOR TYPES
// ============================================================================

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// Aliases for compatibility
typedef half8 half8_t;
typedef half16 half16_t;
typedef float8 float8_t;
typedef float float4_t __attribute__((ext_vector_type(4)));

// ============================================================================
// COMMON CONSTANTS
// ============================================================================

// LDS Padding: 16 + 8 = 24 halfs = 48 bytes
// 48 is not a multiple of 128 bytes (32 banks * 4 bytes), reducing conflicts
#define LDS_PAD 8

// ============================================================================
// PART 1: XOR-BASED LDS SWIZZLE (Original Implementation)
// ============================================================================

namespace lds_swizzle {

// XOR swizzle configuration for bank conflict-free LDS access
// Works by XORing the K-group index with a function of the M index
// This ensures threads accessing same logical column hit different banks

template<int BLOCK_M, int BLOCK_K, int KPACK = 8>
struct XORSwizzle {
    // Number of K-groups (each group is KPACK elements for vectorized access)
    static constexpr int K_GROUPS = BLOCK_K / KPACK;
    
    // Mask for modulo operation (K_GROUPS must be power of 2)
    static constexpr int K_GROUPS_MASK = K_GROUPS - 1;
    
    // Total elements in swizzled buffer (no padding needed!)
    static constexpr int BUFFER_SIZE = BLOCK_M * BLOCK_K;
    
    // ========================================================================
    // Core swizzle function: transforms (row, col) to flat index
    // ========================================================================
    __device__ __forceinline__
    static int to_flat(int row, int col) {
        // Split col into k_group and k_local
        int k_group = col / KPACK;
        int k_local = col % KPACK;
        
        // XOR the k_group with low bits of row
        // This redistributes bank accesses across different rows
        int k_group_swizzled = k_group ^ (row & K_GROUPS_MASK);
        
        // Compute flat index with swizzled k_group
        return row * BLOCK_K + k_group_swizzled * KPACK + k_local;
    }
    
    // ========================================================================
    // Inverse swizzle (for reading) - XOR is self-inverse!
    // ========================================================================
    __device__ __forceinline__
    static int from_flat(int row, int col) {
        // XOR is its own inverse, so same formula works
        return to_flat(row, col);
    }
};

// Specialization for B matrix (stored transposed: [N][K])
template<int BLOCK_N, int BLOCK_K, int KPACK = 8>
struct XORSwizzleB {
    static constexpr int K_GROUPS = BLOCK_K / KPACK;
    static constexpr int K_GROUPS_MASK = K_GROUPS - 1;
    static constexpr int BUFFER_SIZE = BLOCK_N * BLOCK_K;
    
    // For B_lds[n][k], swizzle based on n
    __device__ __forceinline__
    static int to_flat(int n_idx, int k_idx) {
        int k_group = k_idx / KPACK;
        int k_local = k_idx % KPACK;
        int k_group_swizzled = k_group ^ (n_idx & K_GROUPS_MASK);
        return n_idx * BLOCK_K + k_group_swizzled * KPACK + k_local;
    }
};

} // namespace lds_swizzle


// ============================================================================
// PART 2: XOR-BASED LDS SWIZZLE V2 (Row-Based K-Group Swizzle)
// ============================================================================

namespace swizzle_v2 {

// ============================================================================
// SWIZZLE CONFIGURATION
// ============================================================================

// For BLOCK_K=16, we split into 2 groups of 8 (KPACK=8)
// XOR swizzle: k_group' = k_group XOR (row & (NUM_GROUPS-1))
// This creates a diagonal access pattern that avoids bank conflicts

template<int BLOCK_K, int KPACK = 8>
struct RowSwizzle {
    static_assert(BLOCK_K % KPACK == 0, "BLOCK_K must be divisible by KPACK");
    static_assert((BLOCK_K / KPACK) > 0 && ((BLOCK_K / KPACK) & ((BLOCK_K / KPACK) - 1)) == 0,
                  "Number of K-groups must be power of 2");
    
    static constexpr int NUM_GROUPS = BLOCK_K / KPACK;
    static constexpr int GROUP_MASK = NUM_GROUPS - 1;
    
    // Get swizzled K-group index for a given row
    __device__ __forceinline__
    static int swizzle_group(int row, int k_group) {
        return k_group ^ (row & GROUP_MASK);
    }
    
    // Convert logical (row, col) to physical flat index
    // Physical layout: each row has BLOCK_K elements, but K-groups are swizzled
    __device__ __forceinline__
    static int to_physical(int row, int col, int stride) {
        int k_group = col / KPACK;
        int k_local = col % KPACK;
        int swizzled_group = swizzle_group(row, k_group);
        return row * stride + swizzled_group * KPACK + k_local;
    }
    
    // Store a half8 vector at logical position (row, col_start)
    // col_start must be aligned to KPACK (0 or 8 for BLOCK_K=16)
    __device__ __forceinline__
    static void store_vector(__half* lds, int row, int col_start, half8 data, int stride) {
        int k_group = col_start / KPACK;
        int swizzled_group = swizzle_group(row, k_group);
        int physical_idx = row * stride + swizzled_group * KPACK;
        *reinterpret_cast<half8*>(&lds[physical_idx]) = data;
    }
    
    // Load a half8 vector from logical position (row, col_start)
    __device__ __forceinline__
    static half8 load_vector(const __half* lds, int row, int col_start, int stride) {
        int k_group = col_start / KPACK;
        int swizzled_group = swizzle_group(row, k_group);
        int physical_idx = row * stride + swizzled_group * KPACK;
        return *reinterpret_cast<const half8*>(&lds[physical_idx]);
    }
};

} // namespace swizzle_v2


// ============================================================================
// PART 3: L2 CACHE TILE RASTERIZATION
// ============================================================================

namespace tile_rasterization {

// Automatic selection based on matrix shape
enum class RasterMode {
    ROW_MAJOR,
    COLUMN_MAJOR,
    CHUNKED,
    SNAKE,
    DIAGONAL
};

// Simple row-major (baseline - current behavior)
__device__ __forceinline__
void row_major(int& tile_x, int& tile_y, int grid_x, int grid_y) {
    tile_x = blockIdx.x;
    tile_y = blockIdx.y;
}

// Column-major (better for tall matrices)
__device__ __forceinline__
void column_major(int& tile_x, int& tile_y, int grid_x, int grid_y) {
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    tile_x = linear / grid_y;
    tile_y = linear % grid_y;
}

// Chunked column-major (best for L2 locality)
// Processes tiles in small square chunks, column-major within chunks
template<int CHUNK_SIZE = 4>
__device__ __forceinline__
void chunked_column_major(int& tile_x, int& tile_y, int grid_x, int grid_y) {
    // Number of chunks in each dimension
    int chunks_x = (grid_x + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int tiles_per_chunk = CHUNK_SIZE * CHUNK_SIZE;
    
    // Linear tile index
    int linear = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Which chunk and position within chunk
    int chunk_idx = linear / tiles_per_chunk;
    int local_idx = linear % tiles_per_chunk;
    
    // Chunk coordinates
    int chunk_x = chunk_idx % chunks_x;
    int chunk_y = chunk_idx / chunks_x;
    
    // Local coordinates (column-major within chunk)
    int local_x = local_idx / CHUNK_SIZE;
    int local_y = local_idx % CHUNK_SIZE;
    
    // Global tile coordinates
    tile_x = chunk_x * CHUNK_SIZE + local_x;
    tile_y = chunk_y * CHUNK_SIZE + local_y;
    
    // Clamp to valid range
    tile_x = min(tile_x, grid_x - 1);
    tile_y = min(tile_y, grid_y - 1);
}

// Snake pattern (alternating row directions)
__device__ __forceinline__
void snake_pattern(int& tile_x, int& tile_y, int grid_x, int grid_y) {
    tile_y = blockIdx.y;
    if (tile_y & 1) {
        tile_x = grid_x - 1 - blockIdx.x;
    } else {
        tile_x = blockIdx.x;
    }
}

__device__ __forceinline__
void adaptive_rasterize(int& tile_x, int& tile_y, int grid_x, int grid_y, RasterMode mode) {
    switch (mode) {
        case RasterMode::ROW_MAJOR:
            row_major(tile_x, tile_y, grid_x, grid_y);
            break;
        case RasterMode::COLUMN_MAJOR:
            column_major(tile_x, tile_y, grid_x, grid_y);
            break;
        case RasterMode::CHUNKED:
            chunked_column_major<4>(tile_x, tile_y, grid_x, grid_y);
            break;
        case RasterMode::SNAKE:
            snake_pattern(tile_x, tile_y, grid_x, grid_y);
            break;
        default:
            row_major(tile_x, tile_y, grid_x, grid_y);
            break;
    }
}

__host__ inline RasterMode select_raster_mode(int M, int N, int BLOCK_M, int BLOCK_N) {
    int grid_x = (N + BLOCK_N - 1) / BLOCK_N;
    int grid_y = (M + BLOCK_M - 1) / BLOCK_M;
    float aspect = static_cast<float>(grid_y) / static_cast<float>(grid_x);
    
    if (grid_x * grid_y < 64) {
        return RasterMode::ROW_MAJOR;
    } else if (aspect > 2.0f) {
        return RasterMode::COLUMN_MAJOR;
    } else if (aspect < 0.5f) {
        return RasterMode::ROW_MAJOR;
    } else {
        return RasterMode::CHUNKED;
    }
}

} // namespace tile_rasterization


// ============================================================================
// PART 4: SPLIT-K CONFIGURATION AND REDUCTION
// ============================================================================

namespace split_k {

struct SplitKConfig {
    int split_factor;
    int k_per_split;
    size_t workspace_size;
};

__host__ inline SplitKConfig compute_config(
    int M, int N, int K, 
    int BLOCK_M, int BLOCK_N, int BLOCK_K,
    int num_CUs
) {
    SplitKConfig config;
    
    int base_tiles_m = (M + BLOCK_M - 1) / BLOCK_M;
    int base_tiles_n = (N + BLOCK_N - 1) / BLOCK_N;
    int base_tiles = base_tiles_m * base_tiles_n;
    
    int target_tiles = num_CUs * 4;
    
    if (base_tiles >= target_tiles) {
        config.split_factor = 1;
        config.k_per_split = K;
        config.workspace_size = 0;
    } else {
        config.split_factor = (target_tiles + base_tiles - 1) / base_tiles;
        config.split_factor = std::min(config.split_factor, 32);
        
        int min_k_per_split = 256;
        int max_splits = K / min_k_per_split;
        config.split_factor = std::min(config.split_factor, std::max(1, max_splits));
        
        // Round up to power of 2
        int sf = config.split_factor;
        sf--;
        sf |= sf >> 1;
        sf |= sf >> 2;
        sf |= sf >> 4;
        sf++;
        config.split_factor = sf;
        
        config.k_per_split = (K + config.split_factor - 1) / config.split_factor;
        config.workspace_size = (size_t)config.split_factor * M * N * sizeof(float);
    }
    
    return config;
}

template<int BLOCK_SIZE = 256>
__global__ void reduce_kernel(
    float* __restrict__ output,
    const float* __restrict__ partials,
    int M, int N, int split_factor
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total_elements = M * N;
    
    if (idx < total_elements) {
        float sum = 0.0f;
        
        #pragma unroll 4
        for (int s = 0; s < split_factor; s++) {
            sum += partials[s * total_elements + idx];
        }
        
        output[idx] = sum;
    }
}

__host__ inline void launch_reduction(
    float* output,
    const float* partials,
    int M, int N, int split_factor,
    hipStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int total_elements = M * N;
    int grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    reduce_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        output, partials, M, N, split_factor
    );
}

} // namespace split_k


// ============================================================================
// PART 5: OPTIMIZED GEMM KERNEL WITH XOR SWIZZLE (Original Implementation)
// ============================================================================

template<int NWARPS, int WARPS_M, int WARPS_N>
__launch_bounds__(NWARPS * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void wmma_gemm_kernel_xor_swizzle(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K,
    tile_rasterization::RasterMode raster_mode
) {
    // Constants
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = 2 * WMMA_M;
    constexpr int WARP_TILE_N = 2 * WMMA_N;
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;
    constexpr int BLOCK_K = WMMA_K;
    
    // XOR swizzle types - NO PADDING NEEDED
    using SwizzleA = lds_swizzle::XORSwizzle<BLOCK_M, BLOCK_K, 8>;
    using SwizzleB = lds_swizzle::XORSwizzleB<BLOCK_N, BLOCK_K, 8>;
    
    // LDS buffers - compact, no padding waste
    __shared__ __half A_lds[2][SwizzleA::BUFFER_SIZE];
    __shared__ __half B_lds[2][SwizzleB::BUFFER_SIZE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    // Apply tile rasterization for L2 locality
    int tile_x, tile_y;
    int grid_x = (N + BLOCK_N - 1) / BLOCK_N;
    int grid_y = (M + BLOCK_M - 1) / BLOCK_M;
    tile_rasterization::adaptive_rasterize(tile_x, tile_y, grid_x, grid_y, raster_mode);
    
    const int block_m = tile_y * BLOCK_M;
    const int block_n = tile_x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // Accumulators
    float8 c00 = {}, c01 = {}, c10 = {}, c11 = {};
    
    // Load indices for A
    constexpr int A_VECS = BLOCK_M * BLOCK_K / 8;
    const int a_vec_idx = tid;
    const int a_row = a_vec_idx / (BLOCK_K / 8);
    const int a_col = (a_vec_idx % (BLOCK_K / 8)) * 8;
    const bool a_valid = (a_vec_idx < A_VECS) && (block_m + a_row < M);
    
    // Load indices for B (transpose)
    constexpr int B_VECS = BLOCK_K * BLOCK_N / 8;
    const int b_vec_idx = tid;
    const int b_k = b_vec_idx / (BLOCK_N / 8);
    const int b_n = (b_vec_idx % (BLOCK_N / 8)) * 8;
    const bool b_valid = (b_vec_idx < B_VECS) && (b_k < BLOCK_K);
    
    // ========================================================================
    // PROLOGUE: Load first tile with XOR swizzle (vectorized stores)
    // The XOR swizzle: for each row, swap k-groups 0 and 1 based on row parity
    // swap = (row & 1) << 3 gives offset 0 or 8
    // Store k=0..7 at position swap, k=8..15 at position (8 ^ swap)
    // ========================================================================
    if (a_valid && a_col + 8 <= K) {
        half8 data = *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + a_col);
        int swap = (a_row & 1) << 3;
        int store_idx = a_row * BLOCK_K;
        // Store to swizzled position: if a_col is 0..7, store at swap; if 8..15, store at 8^swap
        if (a_col == 0) {
            *reinterpret_cast<half8*>(&A_lds[0][store_idx + swap]) = data;
        } else {
            *reinterpret_cast<half8*>(&A_lds[0][store_idx + (8 ^ swap)]) = data;
        }
    }
    
    if (b_valid && block_n + b_n + 7 < N) {
        half8 data = *reinterpret_cast<const half8*>(B + b_k * N + block_n + b_n);
        // For B transposed: store row-by-row (each b_n index is a row in B_lds)
        // We load 8 consecutive N values but they go into different rows
        // Actually B loading is more complex - each half8 loads 8 consecutive N values
        // which need to be scattered to 8 different rows
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int n_idx = b_n + i;
            int swap_b = (n_idx & 1) << 3;
            int store_idx = n_idx * BLOCK_K;
            // b_k is the k-position (0..15 typically)
            int k_group = b_k >> 3;  // 0 or 1
            int k_local = b_k & 7;
            int phys_k = (k_group == 0) ? (swap_b + k_local) : ((8 ^ swap_b) + k_local);
            B_lds[0][store_idx + phys_k] = reinterpret_cast<__half*>(&data)[i];
        }
    }
    
    __syncthreads();
    
    int curr_buf = 0;
    const int frag_col = lane_id % 16;
    
    // Pointers for prefetch
    const __half* A_ptr = A + (block_m + a_row) * K;
    const __half* B_base = B + block_n + b_n;
    
    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    #pragma unroll 1
    for (int k = 0; k < K; k += BLOCK_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + BLOCK_K < K);
        
        // Load fragments from XOR-swizzled LDS using vectorized loads
        // This matches the proven-good pattern from load_matrix_sync_lds_swizzled()
        // Key insight: Each lane loads its own ROW (lane % 16), getting 16 K-elements
        // The XOR swizzle reversal is done inline via swap offset
        half16 a0, a1, b0, b1;
        
        const int lane_row = lane_id & 15;  // Which row this lane loads (0-15)
        
        // ========================================================================
        // A Fragment Loading (vectorized, with inline XOR swizzle reversal)
        // Each lane loads one row of the 16x16 tile
        // ========================================================================
        {
            // For tile a0: rows [warp_m_base .. warp_m_base+15]
            int a0_row = warp_m_base + lane_row;
            const __half* a0_row_ptr = &A_lds[curr_buf][a0_row * BLOCK_K];
            
            // XOR swizzle reversal: swap = (row & 1) << 3, gives 0 or 8
            int swap0 = (a0_row & 1) << 3;
            half8 va0_0 = *reinterpret_cast<const half8*>(a0_row_ptr + swap0);
            half8 va0_1 = *reinterpret_cast<const half8*>(a0_row_ptr + (8 ^ swap0));
            
            // Unpack into a0 fragment in logical order (k=0..7 then k=8..15)
            #pragma unroll
            for (int i = 0; i < 8; i++) a0[i]     = va0_0[i];
            #pragma unroll
            for (int i = 0; i < 8; i++) a0[i + 8] = va0_1[i];
            
            // For tile a1: rows [warp_m_base+16 .. warp_m_base+31]
            int a1_row = warp_m_base + 16 + lane_row;
            const __half* a1_row_ptr = &A_lds[curr_buf][a1_row * BLOCK_K];
            
            int swap1 = (a1_row & 1) << 3;
            half8 va1_0 = *reinterpret_cast<const half8*>(a1_row_ptr + swap1);
            half8 va1_1 = *reinterpret_cast<const half8*>(a1_row_ptr + (8 ^ swap1));
            
            #pragma unroll
            for (int i = 0; i < 8; i++) a1[i]     = va1_0[i];
            #pragma unroll
            for (int i = 0; i < 8; i++) a1[i + 8] = va1_1[i];
        }
        
        // ========================================================================
        // B Fragment Loading (vectorized, with inline XOR swizzle reversal)
        // B is transposed in LDS as B_lds[n][k], each lane loads one column (n = lane_row)
        // ========================================================================
        {
            // For tile b0: columns [warp_n_base .. warp_n_base+15]
            int b0_col = warp_n_base + lane_row;
            const __half* b0_col_ptr = &B_lds[curr_buf][b0_col * BLOCK_K];
            
            // XOR swizzle for B uses column parity
            int swapb0 = (b0_col & 1) << 3;
            half8 vb0_0 = *reinterpret_cast<const half8*>(b0_col_ptr + swapb0);
            half8 vb0_1 = *reinterpret_cast<const half8*>(b0_col_ptr + (8 ^ swapb0));
            
            #pragma unroll
            for (int i = 0; i < 8; i++) b0[i]     = vb0_0[i];
            #pragma unroll
            for (int i = 0; i < 8; i++) b0[i + 8] = vb0_1[i];
            
            // For tile b1: columns [warp_n_base+16 .. warp_n_base+31]
            int b1_col = warp_n_base + 16 + lane_row;
            const __half* b1_col_ptr = &B_lds[curr_buf][b1_col * BLOCK_K];
            
            int swapb1 = (b1_col & 1) << 3;
            half8 vb1_0 = *reinterpret_cast<const half8*>(b1_col_ptr + swapb1);
            half8 vb1_1 = *reinterpret_cast<const half8*>(b1_col_ptr + (8 ^ swapb1));
            
            #pragma unroll
            for (int i = 0; i < 8; i++) b1[i]     = vb1_0[i];
            #pragma unroll
            for (int i = 0; i < 8; i++) b1[i + 8] = vb1_1[i];
        }

        
        // Prefetch next tile using same swizzled store pattern as prologue
        if (has_next) {
            int next_k = k + BLOCK_K;
            if (a_valid && next_k + a_col + 8 <= K) {
                half8 data = *reinterpret_cast<const half8*>(A_ptr + next_k + a_col);
                int swap = (a_row & 1) << 3;
                int store_idx = a_row * BLOCK_K;
                if (a_col == 0) {
                    *reinterpret_cast<half8*>(&A_lds[next_buf][store_idx + swap]) = data;
                } else {
                    *reinterpret_cast<half8*>(&A_lds[next_buf][store_idx + (8 ^ swap)]) = data;
                }
            }
            
            if (b_valid && block_n + b_n + 7 < N) {
                half8 data = *reinterpret_cast<const half8*>(B_base + (next_k + b_k) * N);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int n_idx = b_n + i;
                    int swap_b = (n_idx & 1) << 3;
                    int store_idx = n_idx * BLOCK_K;
                    int k_group = b_k >> 3;
                    int k_local = b_k & 7;
                    int phys_k = (k_group == 0) ? (swap_b + k_local) : ((8 ^ swap_b) + k_local);
                    B_lds[next_buf][store_idx + phys_k] = reinterpret_cast<__half*>(&data)[i];
                }
            }
        }

        
        // WMMA compute
        if (warp_active) {
            c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
            c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
            c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
            c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // ========================================================================
    // EPILOGUE: Store results
    // 
    // WMMA fragment layout (from docs/wmma_fragment_layout_rdna3.md):
    // - Fragment has 8 float32 elements (c_frag[0..7])
    // - Lane L stores to column: L % 16 (fixed per lane)
    // - Element c_frag[i] stores to row: i*2 + (L / 16)
    //   - Lanes 0-15: rows 0,2,4,...,14 (even rows)
    //   - Lanes 16-31: rows 1,3,5,...,15 (odd rows)
    // ========================================================================
    if (!warp_active) return;
    
    // frag_col is already defined above at line 392, reuse it
    const int frag_row_offset = lane_id / 16;    // 0 for lanes 0-15, 1 for lanes 16-31
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // Each fragment element i maps to row i*2 + row_offset in the 16x16 tile
        int frag_row = i * 2 + frag_row_offset;  // Rows: 0,2,4,...,14 or 1,3,5,...,15
        
        // Global row/column coordinates
        int gr0 = block_m + warp_m_base + frag_row;
        int gc0 = block_n + warp_n_base + frag_col;
        
        // Tile [0][0]: rows 0-15, columns 0-15
        if (gr0 < M && gc0 < N) C[gr0 * N + gc0] = c00[i];
        
        // Tile [0][1]: rows 0-15, columns 16-31
        int gc1 = gc0 + 16;
        if (gr0 < M && gc1 < N) C[gr0 * N + gc1] = c01[i];
        
        // Tile [1][0]: rows 16-31, columns 0-15
        int gr1 = gr0 + 16;
        if (gr1 < M && gc0 < N) C[gr1 * N + gc0] = c10[i];
        
        // Tile [1][1]: rows 16-31, columns 16-31
        if (gr1 < M && gc1 < N) C[gr1 * N + gc1] = c11[i];
    }
}


// ============================================================================
// PART 6: SPLIT-K GEMM KERNEL
// ============================================================================

template<int NWARPS, int WARPS_M, int WARPS_N>
__launch_bounds__(NWARPS * 32, 2)
__global__ void wmma_gemm_kernel_split_k(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ workspace,
    const int M, const int N, const int K,
    const int split_factor,
    const int k_per_split
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_TILE_M = 2 * WMMA_M;
    constexpr int WARP_TILE_N = 2 * WMMA_N;
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;
    constexpr int BLOCK_K = WMMA_K;
    constexpr int LOCAL_LDS_PAD = 8;
    constexpr int A_STRIDE = BLOCK_K + LOCAL_LDS_PAD;
    constexpr int B_STRIDE = BLOCK_K + LOCAL_LDS_PAD;
    
    const int split_idx = blockIdx.z;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    
    const int block_m = tile_m * BLOCK_M;
    const int block_n = tile_n * BLOCK_N;
    
    const int k_start = split_idx * k_per_split;
    const int k_end = min(k_start + k_per_split, K);
    
    if (block_m >= M || block_n >= N || k_start >= K) return;
    
    __shared__ __half A_lds[2][BLOCK_M][A_STRIDE];
    __shared__ __half B_lds[2][BLOCK_N][B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    float8 c00 = {}, c01 = {}, c10 = {}, c11 = {};
    
    const int a_row = tid / 2;
    const int a_col = (tid % 2) * 8;
    const bool a_valid = (a_row < BLOCK_M) && (block_m + a_row < M);
    
    const int b_k = tid / 8;
    const int b_n = (tid % 8) * 8;
    const bool b_valid = (b_k < BLOCK_K) && (block_n + b_n + 7 < N);
    
    // Prologue
    int k = k_start;
    if (a_valid && k + a_col + 8 <= K) {
        half8 data = *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + k + a_col);
        *reinterpret_cast<half8*>(&A_lds[0][a_row][a_col]) = data;
    }
    if (b_valid && k + b_k < K) {
        half8 data = *reinterpret_cast<const half8*>(B + (k + b_k) * N + block_n + b_n);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            B_lds[0][b_n + i][b_k] = reinterpret_cast<__half*>(&data)[i];
        }
    }
    __syncthreads();
    
    int curr_buf = 0;
    const int frag_col = lane_id % 16;
    
    #pragma unroll 1
    for (; k < k_end; k += BLOCK_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + BLOCK_K < k_end);
        
        half16 a0, a1, b0, b1;
        
        #pragma unroll
        for (int row = 0; row < 16; row++) {
            a0[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + row][frag_col]);
            a1[row] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][warp_m_base + 16 + row][frag_col]);
        }
        
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + frag_col][kk]);
            b1[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][warp_n_base + 16 + frag_col][kk]);
        }
        
        if (has_next) {
            const int next_k = k + BLOCK_K;
            if (a_valid && next_k + a_col + 8 <= K) {
                half8 data = *reinterpret_cast<const half8*>(A + (block_m + a_row) * K + next_k + a_col);
                *reinterpret_cast<half8*>(&A_lds[next_buf][a_row][a_col]) = data;
            }
            if (b_valid && next_k + b_k < K) {
                half8 data = *reinterpret_cast<const half8*>(B + (next_k + b_k) * N + block_n + b_n);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    B_lds[next_buf][b_n + i][b_k] = reinterpret_cast<__half*>(&data)[i];
                }
            }
        }
        
        if (warp_active) {
            c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
            c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
            c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
            c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        }
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    if (!warp_active) return;
    
    // WMMA fragment layout: c_frag[i] -> row = i*2 + (lane/16), col = lane%16
    const size_t split_offset = (size_t)split_idx * M * N;
    float* out = workspace + split_offset;
    
    // frag_col is already defined above, reuse it
    const int frag_row_offset = lane_id / 16;    // 0 for lanes 0-15, 1 for lanes 16-31
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // Each fragment element i maps to row i*2 + row_offset in the 16x16 tile
        int frag_row = i * 2 + frag_row_offset;  // Rows: 0,2,4,...,14 or 1,3,5,...,15
        
        // Global row/column coordinates
        int gr0 = block_m + warp_m_base + frag_row;
        int gc0 = block_n + warp_n_base + frag_col;
        
        // Tile [0][0]: rows 0-15, columns 0-15
        if (gr0 < M && gc0 < N) out[gr0 * N + gc0] = c00[i];
        
        // Tile [0][1]: rows 0-15, columns 16-31
        int gc1 = gc0 + 16;
        if (gr0 < M && gc1 < N) out[gr0 * N + gc1] = c01[i];
        
        // Tile [1][0]: rows 16-31, columns 0-15
        int gr1 = gr0 + 16;
        if (gr1 < M && gc0 < N) out[gr1 * N + gc0] = c10[i];
        
        // Tile [1][1]: rows 16-31, columns 16-31
        if (gr1 < M && gc1 < N) out[gr1 * N + gc1] = c11[i];
    }
}


// ============================================================================
// PART 7: CORRECTED SWIZZLED KERNEL V2 (Row-Based K-Group Swizzle)
// ============================================================================
//
// Key insight: Fragment loading requires CONTIGUOUS access within each
// K-slice. So we can't swizzle individual elements - we must swizzle
// entire K-groups while keeping elements within groups contiguous.
//
// For fragment loading:
// - Each lane needs elements from a specific COLUMN of A
// - We load 16 elements (one column) which spans both K-groups
// - We must un-swizzle when reading to get correct logical order
// ============================================================================

template<int NWARPS, int WARPS_M_PARAM, int WARPS_N_PARAM>
__launch_bounds__(NWARPS * 32, 2)
__attribute__((amdgpu_waves_per_eu(4, 8)))
__global__ void wmma_gemm_kernel_xor_swizzle_v2(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_SIZE = 32;
    
    constexpr int WARPS_M = WARPS_M_PARAM;
    constexpr int WARPS_N = WARPS_N_PARAM;
    constexpr int WARP_TILE_M = 2 * WMMA_M;  // 32
    constexpr int WARP_TILE_N = 2 * WMMA_N;  // 32
    
    constexpr int BLOCK_M = WARPS_M * WARP_TILE_M;  // 128 for 4 warps in M
    constexpr int BLOCK_N = WARPS_N * WARP_TILE_N;  // 64 for 2 warps in N
    constexpr int BLOCK_K = WMMA_K;                  // 16
    
    // NO PADDING - XOR swizzle eliminates bank conflicts
    constexpr int A_STRIDE = BLOCK_K;  // 16 (was 24 with padding)
    constexpr int B_STRIDE = BLOCK_K;  // 16 (was 24 with padding)
    
    using Swizzle = swizzle_v2::RowSwizzle<BLOCK_K, 8>;
    
    // Flat LDS arrays
    __shared__ __half A_lds[2][BLOCK_M * A_STRIDE];
    __shared__ __half B_lds[2][BLOCK_N * B_STRIDE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_m_base = warp_m * WARP_TILE_M;
    const int warp_n_base = warp_n * WARP_TILE_N;
    
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    
    if (block_m >= M || block_n >= N) return;
    
    const bool warp_active = (block_m + warp_m_base < M) && (block_n + warp_n_base < N);
    
    // Accumulators
    float8 c00 = {}, c01 = {}, c10 = {}, c11 = {};
    
    // ========================================================================
    // LOAD ASSIGNMENTS
    // A: BLOCK_M × BLOCK_K = 128 × 16 = 2048 halfs = 256 half8 vectors
    // B: BLOCK_K × BLOCK_N = 16 × 64 = 1024 halfs = 128 half8 vectors
    // 256 threads can handle A exactly, B uses first 128 threads
    // ========================================================================
    
    // A load: each thread loads one half8 (row, cols 0-7 or 8-15)
    constexpr int A_VECS = BLOCK_M * BLOCK_K / 8;  // 256
    const int a_vec_idx = tid;
    const int a_row = a_vec_idx / (BLOCK_K / 8);   // 0..127
    const int a_col = (a_vec_idx % (BLOCK_K / 8)) * 8;  // 0 or 8
    const bool a_valid = (a_vec_idx < A_VECS) && (block_m + a_row < M);
    
    // B load: each thread loads one half8, then scatter-writes to transpose
    // B is [K][N] row-major, we want B_lds[N][K] for col_major fragment
    constexpr int B_VECS = BLOCK_K * BLOCK_N / 8;  // 128
    const int b_vec_idx = tid;
    const int b_k = b_vec_idx / (BLOCK_N / 8);     // 0..15 (K row)
    const int b_n = (b_vec_idx % (BLOCK_N / 8)) * 8;  // 0..56 (N col start)
    const bool b_valid = (b_vec_idx < B_VECS) && (block_n + b_n + 7 < N);
    
    // Base pointers
    const __half* A_ptr = A + (block_m + a_row) * K + a_col;
    const __half* B_ptr = B + b_k * N + block_n + b_n;
    
    // ========================================================================
    // PROLOGUE: Load first tile with swizzle
    // ========================================================================
    if (a_valid && a_col + 8 <= K) {
        half8 data = *reinterpret_cast<const half8*>(A_ptr);
        // Store with row-based XOR swizzle
        Swizzle::store_vector(A_lds[0], a_row, a_col, data, A_STRIDE);
    }
    
    if (b_valid && b_k < K) {
        half8 data = *reinterpret_cast<const half8*>(B_ptr);
        // Transpose: B[k][n..n+7] -> B_lds[n][k], B_lds[n+1][k], ...
        // Each of the 8 elements goes to a different row in B_lds
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int n_idx = b_n + i;
            // Apply swizzle to the transposed storage
            int phys_idx = Swizzle::to_physical(n_idx, b_k, B_STRIDE);
            B_lds[0][phys_idx] = reinterpret_cast<__half*>(&data)[i];
        }
    }
    
    __syncthreads();
    
    int curr_buf = 0;
    
    // Fragment index: for RDNA3 WMMA, lanes 0-15 and 16-31 need same data
    const int frag_col = lane_id % 16;
    
    // ========================================================================
    // MAIN LOOP
    // ========================================================================
    #pragma unroll 1
    for (int k = 0; k < K; k += BLOCK_K) {
        const int next_buf = 1 - curr_buf;
        const bool has_next = (k + BLOCK_K < K);
        
        // ====================================================================
        // LOAD FRAGMENTS FROM SWIZZLED LDS
        // 
        // For A (row-major): fragment needs column frag_col, rows 0..15
        // Each row may have its K-groups swizzled differently
        // We must un-swizzle each row independently
        //
        // For B (transposed to col-major): fragment needs row frag_col, 
        // all K values 0..15
        // ====================================================================
        
        half16 a0, a1, b0, b1;
        
        // Load A fragments: 2 tiles of 16x16 each
        // Tile 0: rows [warp_m_base .. warp_m_base+15], column frag_col
        // Tile 1: rows [warp_m_base+16 .. warp_m_base+31], column frag_col
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            int row0 = warp_m_base + r;
            int row1 = warp_m_base + 16 + r;
            
            // Un-swizzle: get physical location of logical (row, frag_col)
            int phys0 = Swizzle::to_physical(row0, frag_col, A_STRIDE);
            int phys1 = Swizzle::to_physical(row1, frag_col, A_STRIDE);
            
            a0[r] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][phys0]);
            a1[r] = *reinterpret_cast<const _Float16*>(&A_lds[curr_buf][phys1]);
        }
        
        // Load B fragments: 2 tiles of 16x16 each
        // Tile 0: row (warp_n_base + frag_col), columns 0..15
        // Tile 1: row (warp_n_base + 16 + frag_col), columns 0..15
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            int n0 = warp_n_base + frag_col;
            int n1 = warp_n_base + 16 + frag_col;
            
            // Un-swizzle
            int phys0 = Swizzle::to_physical(n0, kk, B_STRIDE);
            int phys1 = Swizzle::to_physical(n1, kk, B_STRIDE);
            
            b0[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][phys0]);
            b1[kk] = *reinterpret_cast<const _Float16*>(&B_lds[curr_buf][phys1]);
        }
        
        // ====================================================================
        // PREFETCH NEXT TILE
        // ====================================================================
        half8 a_prefetch = {};
        half8 b_prefetch = {};
        
        if (has_next) {
            if (a_valid && (k + BLOCK_K + a_col + 8 <= K)) {
                a_prefetch = *reinterpret_cast<const half8*>(A_ptr + BLOCK_K);
            }
            if (b_valid && (k + BLOCK_K + b_k < K)) {
                b_prefetch = *reinterpret_cast<const half8*>(B_ptr + BLOCK_K * N);
            }
        }
        
        // ====================================================================
        // WMMA COMPUTE
        // ====================================================================
        if (warp_active) {
            c00 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, c00);
            c01 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b1, c01);
            c10 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b0, c10);
            c11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, c11);
        }
        
        // ====================================================================
        // WRITE PREFETCH TO NEXT BUFFER (with swizzle)
        // ====================================================================
        if (has_next) {
            if (a_valid && (k + BLOCK_K + a_col + 8 <= K)) {
                Swizzle::store_vector(A_lds[next_buf], a_row, a_col, a_prefetch, A_STRIDE);
            }
            
            if (b_valid && (k + BLOCK_K + b_k < K)) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int n_idx = b_n + i;
                    int phys_idx = Swizzle::to_physical(n_idx, b_k, B_STRIDE);
                    B_lds[next_buf][phys_idx] = reinterpret_cast<__half*>(&b_prefetch)[i];
                }
            }
        }
        
        A_ptr += BLOCK_K;
        B_ptr += BLOCK_K * N;
        
        __syncthreads();
        curr_buf = next_buf;
    }
    
    // ========================================================================
    // EPILOGUE: Store results
    // ========================================================================
    if (!warp_active) return;
    
    // Fragment layout for RDNA3 WMMA:
    // - Lane i (0-31) holds elements at row positions based on lane_id
    // - lane_id % 16 gives column offset
    // - lane_id / 16 gives row offset (0 or 1)
    // - Each lane has 8 elements covering rows 0,2,4,6,8,10,12,14 (or +1)
    
    const int frag_row_offset = lane_id / 16;  // 0 or 1
    const int frag_col_offset = lane_id % 16;  // 0-15
    
    #pragma unroll
    for (int ti = 0; ti < 2; ti++) {  // Tile in M direction
        #pragma unroll
        for (int tj = 0; tj < 2; tj++) {  // Tile in N direction
            const int tile_row_base = block_m + warp_m_base + ti * WMMA_M;
            const int tile_col_base = block_n + warp_n_base + tj * WMMA_N;
            
            // Select correct accumulator
            const float8& acc = (ti == 0) ? ((tj == 0) ? c00 : c01) 
                                          : ((tj == 0) ? c10 : c11);
            
            const int c_col = tile_col_base + frag_col_offset;
            
            if (c_col < N) {
                #pragma unroll
                for (int elem = 0; elem < 8; elem++) {
                    // Each element covers row: elem*2 + frag_row_offset
                    const int c_row = tile_row_base + elem * 2 + frag_row_offset;
                    if (c_row < M) {
                        C[c_row * N + c_col] = acc[elem];
                    }
                }
            }
        }
    }
}

// Template instantiation
template __global__ void wmma_gemm_kernel_xor_swizzle_v2<8, 4, 2>(
    const __half*, const __half*, float*, int, int, int);


// ============================================================================
// PART 8: HOST WRAPPER FUNCTIONS
// ============================================================================

inline void wmma_gemm_xor_swizzle(
    const __half* A, const __half* B, float* C,
    int M, int N, int K,
    hipStream_t stream = 0
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    
    // Use ROW_MAJOR for now to match standard kernel behavior
    // TODO: Enable adaptive rasterization once correctness is verified
    auto mode = tile_rasterization::RasterMode::ROW_MAJOR;
    // auto mode = tile_rasterization::select_raster_mode(M, N, BLOCK_M, BLOCK_N);
    
    wmma_gemm_kernel_xor_swizzle<8, 4, 2><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K, mode
    );
}

inline void wmma_gemm_split_k_wrapper(
    const __half* A, const __half* B, float* C,
    int M, int N, int K,
    float* workspace,
    int split_factor,
    hipStream_t stream = 0
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    
    int k_per_split = (K + split_factor - 1) / split_factor;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, 
              (M + BLOCK_M - 1) / BLOCK_M, 
              split_factor);
    dim3 block(256);
    
    wmma_gemm_kernel_split_k<8, 4, 2><<<grid, block, 0, stream>>>(
        A, B, workspace, M, N, K, split_factor, k_per_split
    );
    
    split_k::launch_reduction(C, workspace, M, N, split_factor, stream);
}

inline void wmma_gemm_adaptive_optimized(
    const __half* A, const __half* B, float* C,
    int M, int N, int K,
    hipStream_t stream = 0
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    constexpr int NUM_CUS = 40;
    
    auto sk_config = split_k::compute_config(M, N, K, BLOCK_M, BLOCK_N, 16, NUM_CUS);
    
    if (sk_config.split_factor > 1) {
        float* workspace;
        hipMalloc(&workspace, sk_config.workspace_size);
        
        wmma_gemm_split_k_wrapper(A, B, C, M, N, K, workspace, 
                                   sk_config.split_factor, stream);
        
        hipStreamSynchronize(stream);
        hipFree(workspace);
    } else {
        wmma_gemm_xor_swizzle(A, B, C, M, N, K, stream);
    }
}

inline void wmma_gemm_xor_swizzle_v2(
    const __half* A, const __half* B, float* C,
    int M, int N, int K,
    hipStream_t stream = 0
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    
    wmma_gemm_kernel_xor_swizzle_v2<8, 4, 2><<<grid, block, 0, stream>>>(
        A, B, C, M, N, K
    );
}

// ============================================================================
// PART 9: EPILOGUE OPTIMIZATION (Vectorized LDS-Buffered Writes)
// ============================================================================
//
// Optimization: Instead of writing WMMA fragments directly to global memory
// (one element at a time with scatter pattern), we:
// 1. Collect fragments into shared memory
// 2. Sync threads
// 3. Write from LDS to global using coalesced vectorized stores (float8)
//
// Based on: adelj_wmma_samples/hgemm/src/wmma_opt_5.cpp lines 227-319
// ============================================================================

namespace epilogue_opt {

// Fragment layout constants for RDNA3 WMMA FP32 output
// - Lane L stores to column: L % 16
// - Element c_frag[i] stores to row: i*2 + (L / 16)
constexpr int FRAG_ELEMENTS = 8;

/**
 * @brief Store WMMA FP32 fragments to LDS for coalesced writeback
 * 
 * @param c_lds       Shared memory buffer for C tile (must hold BLOCK_M * BLOCK_N floats)
 * @param c_frag      Pointer to 4 float8 accumulators (c00, c01, c10, c11)
 * @param warp_m_base Starting M coordinate for this warp
 * @param warp_n_base Starting N coordinate for this warp
 * @param lane_id     Lane ID within warp (0-31)
 * @param BLOCK_N     Block size in N dimension (stride)
 */
template<int BLOCK_N>
__device__ __forceinline__ void store_fragments_to_lds(
    float* c_lds,
    const float8* c_frags,  // Array of 4: c00, c01, c10, c11
    int warp_m_base,
    int warp_n_base,
    int lane_id
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    
    const int frag_col = lane_id % 16;
    const int frag_row_offset = lane_id / 16;  // 0 or 1
    
    #pragma unroll
    for (int ti = 0; ti < 2; ti++) {
        #pragma unroll
        for (int tj = 0; tj < 2; tj++) {
            const int tile_m = ti * WMMA_M;
            const int tile_n = tj * WMMA_N;
            const int frag_idx = ti * 2 + tj;
            const float8& frag = c_frags[frag_idx];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int row = warp_m_base + tile_m + i * 2 + frag_row_offset;
                int col = warp_n_base + tile_n + frag_col;
                c_lds[row * BLOCK_N + col] = frag[i];
            }
        }
    }
}

/**
 * @brief Vectorized LDS to global memory write
 * 
 * Uses float8 (256-bit) vectorized writes for optimal memory bandwidth.
 * All threads cooperatively write the entire C tile.
 */
template<int BLOCK_M, int BLOCK_N, int NTHREADS>
__device__ __forceinline__ void vectorized_lds_to_global(
    float* C,
    const float* c_lds,
    int block_m,
    int block_n,
    int M, int N,
    int tid
) {
    constexpr int VECTOR_WIDTH = 8;  // float8
    constexpr int TOTAL_ELEMENTS = BLOCK_M * BLOCK_N;
    constexpr int VECTORS_PER_THREAD = (TOTAL_ELEMENTS / VECTOR_WIDTH + NTHREADS - 1) / NTHREADS;
    
    #pragma unroll
    for (int v = 0; v < VECTORS_PER_THREAD; v++) {
        int vec_idx = tid + v * NTHREADS;
        int elem_idx = vec_idx * VECTOR_WIDTH;
        
        if (elem_idx >= TOTAL_ELEMENTS) break;
        
        int row_local = elem_idx / BLOCK_N;
        int col_local = elem_idx % BLOCK_N;
        int row_global = block_m + row_local;
        int col_global = block_n + col_local;
        
        // Check if entire vector is in bounds
        if (row_global < M && col_global + VECTOR_WIDTH - 1 < N) {
            // Full vectorized write
            *reinterpret_cast<float8*>(&C[row_global * N + col_global]) =
                *reinterpret_cast<const float8*>(&c_lds[elem_idx]);
        } else if (row_global < M) {
            // Boundary case: write element by element
            #pragma unroll
            for (int i = 0; i < VECTOR_WIDTH; i++) {
                if (col_global + i < N && elem_idx + i < TOTAL_ELEMENTS) {
                    C[row_global * N + col_global + i] = c_lds[elem_idx + i];
                }
            }
        }
    }
}

} // namespace epilogue_opt

#endif // WMMA_XOR_SWIZZLE_HPP


