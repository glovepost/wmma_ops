/*
 * rocWMMA patch for gfx1151 (RDNA3.5) - wave32
 * WMMA: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
 *
 * ABI (gfx11): float8 = wmma(half16 a, half16 b, float8 c)
 * See: https://codebrowser.dev/llvm/clang/include/clang/Basic/BuiltinsAMDGPU.def.html
 *
 * Lane mapping (wave32):
 *  - lane_mod = lane % 16
 *  - A fragment: lane loads row lane_mod (16 elements)
 *  - B fragment: lane loads column lane_mod (16 elements), i.e. col-major fragment
 *  - Accum fragment: 8 floats per lane; lane/16 selects even/odd rows for that column
 *
 * Swizzle (KPACK=8, WMMA_K=16 => KG=2):
 *  - For A rows: swap [0..7] and [8..15] groups on odd rows to avoid LDS bank conflicts.
 *  - For B (transposed into LDS as B_lds[n][k]): swap K-groups on odd n (column index).
 */

#ifndef ROCWMMA_GFX1151_PATCH_HPP
#define ROCWMMA_GFX1151_PATCH_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <type_traits>
#include <stdint.h>

namespace rocwmma {

constexpr int WAVE_SIZE = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

struct row_major {};
struct col_major {};
struct mem_row_major {};
struct mem_col_major {};

struct matrix_a {};
struct matrix_b {};
struct accumulator {};

template<typename FragType, int M, int N, int K, typename DataT, typename Layout = row_major>
class fragment {
public:
    static constexpr int num_elements =
        (std::is_same<FragType, accumulator>::value) ? (M * N / WAVE_SIZE) : 16;
    DataT x[num_elements];
    __device__ __forceinline__ DataT& operator[](int i) { return x[i]; }
    __device__ __forceinline__ const DataT& operator[](int i) const { return x[i]; }
};

template<typename FragType, int M, int N, int K, typename DataT, typename Layout>
__device__ __forceinline__ void fill_fragment(fragment<FragType, M, N, K, DataT, Layout>& frag, DataT val) {
    #pragma unroll
    for (int i = 0; i < frag.num_elements; i++) frag.x[i] = val;
}

// ---------------------------
// Vector types (device)
// ---------------------------
typedef _Float16 f16;
typedef f16 half16_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));
typedef f16 half8_t __attribute__((ext_vector_type(8)));

// ---------------------------
// Bitcast helpers (NO CONVERSION, just reinterpret bits)
// ---------------------------
__device__ __forceinline__ f16 bitcast_f16(__half h) {
#if defined(__clang__)
    return __builtin_bit_cast(f16, h);
#else
    // fallback: relies on identical bit layout
    return *reinterpret_cast<f16*>(&h);
#endif
}

__device__ __forceinline__ __half bitcast_half(f16 h) {
#if defined(__clang__)
    return __builtin_bit_cast(__half, h);
#else
    return *reinterpret_cast<__half*>(&h);
#endif
}

// ---------------------------
// XOR swizzle helpers (KPACK=8, KG=2)
// For WMMA_K=16, there are exactly 2 K-groups.
// Swizzle is a simple swap of groups based on row/col parity.
// ---------------------------
constexpr int KPACK = 8;
constexpr int KG = WMMA_K / KPACK; // 2
static_assert(KG == 2, "This swizzle implementation assumes WMMA_K=16");

// Swap group index based on parity bit
__device__ __forceinline__ int swz_group(int group, int xor_bit) {
    // group in {0,1}, xor_bit in {0,1}
    return group ^ (xor_bit & 1);
}

// For A row-major stored as A_lds[row][phys_k]
// Returns physical k index for logical k
__device__ __forceinline__ int a_phys_k(int row, int k) {
    const int group = k >> 3;         // 0 or 1
    const int local = k & 7;          // 0..7
    const int pg = swz_group(group, row);
    return (pg << 3) | local;
}

// For B transposed stored as B_lds[n][phys_k]
// k dimension swizzled by n (column) parity
__device__ __forceinline__ int b_phys_k(int n, int k) {
    const int group = k >> 3;         // 0 or 1
    const int local = k & 7;          // 0..7
    const int pg = swz_group(group, n);
    return (pg << 3) | local;
}

// ---------------------------
// Matrix A loaders (row-major fragment)
// lane loads row = lane%16
// ---------------------------

// Load A from swizzled LDS A_lds[row][phys_k]
// Use this with swizzled LDS stores (no padding needed)
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds_swizzled(
    fragment<matrix_a, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,   // points to A_lds[tile_row][0]
    int ldm                   // physical stride in LDS (>=16). For swizzled no-pad, ldm==16.
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int row = lane & 15;

    const __half* row_ptr = base_ptr + row * ldm;

    // For KPACK=8, KG=2: physical groups are swapped based on row parity
    // Inline XOR: phys0 = (row & 1) ? 8 : 0, phys1 = (row & 1) ? 0 : 8
    const int swap = (row & 1) << 3;  // 0 or 8
    
    // Load half8 twice
    const half8_t v0 = *reinterpret_cast<const half8_t*>(row_ptr + swap);
    const half8_t v1 = *reinterpret_cast<const half8_t*>(row_ptr + (8 ^ swap));

    // Place into fragment in logical order
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
}

// ---------------------------
// Matrix B loaders (col-major fragment) - SWIZZLED version
// lane loads col = lane%16, needs 16 elements down k dimension.
// B is stored in LDS transposed as B_lds[n][phys_k].
// ---------------------------
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds_b_transposed_swizzled(
    fragment<matrix_b, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,   // points to B_lds[tile_col][0]
    int col_stride            // physical stride (>=16). For swizzled no-pad, col_stride==16.
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int col = lane & 15;

    const __half* col_ptr = base_ptr + col * col_stride;

    // Logical k=0..15, but physical k swizzled by col parity.
    const int phys0 = swz_group(0, col) << 3;
    const int phys1 = swz_group(1, col) << 3;

    const half8_t v0 = *reinterpret_cast<const half8_t*>(col_ptr + phys0);
    const half8_t v1 = *reinterpret_cast<const half8_t*>(col_ptr + phys1);

    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
}

// ---------------------------
// Standard loaders for padded LDS layout (DEFAULT - backward compatible)
// Use with A_lds[row][col] and B_lds[n][k] with LDS_PAD
// ---------------------------
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds(
    fragment<matrix_a, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,
    int ldm
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int row = lane & 15;
    const __half* row_ptr = base_ptr + row * ldm;
    
    // Direct load without swizzle accounting
    const half8_t v0 = *reinterpret_cast<const half8_t*>(row_ptr);
    const half8_t v1 = *reinterpret_cast<const half8_t*>(row_ptr + 8);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
}

template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync_lds_b_transposed(
    fragment<matrix_b, M, N, K, __half, Layout>& frag,
    const __half* base_ptr,
    int col_stride
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int col = lane & 15;
    const __half* col_ptr = base_ptr + col * col_stride;
    
    const half8_t v0 = *reinterpret_cast<const half8_t*>(col_ptr);
    const half8_t v1 = *reinterpret_cast<const half8_t*>(col_ptr + 8);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i]     = bitcast_half(v0[i]);
    #pragma unroll
    for (int i = 0; i < 8; i++) frag.x[i + 8] = bitcast_half(v1[i]);
}

// ---------------------------
// WMMA wrapper (NO FLOAT ROUND-TRIP - uses bitcast)
// ---------------------------
template<typename FragA, typename FragB, typename FragC>
__device__ __forceinline__ void mma_sync(FragC& d, const FragA& a, const FragB& b, const FragC& c) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__AMDGCN__)
    half16_t a_vec, b_vec;
    float8_t c_vec;

    // Pack A and B fragments using bitcast (no numeric conversion!)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        a_vec[i] = bitcast_f16(a.x[i]);
        b_vec[i] = bitcast_f16(b.x[i]);
    }
    
    // Pack accumulator
    #pragma unroll
    for (int i = 0; i < 8; i++) c_vec[i] = c.x[i];

    // Intrinsic signature: float8 = wmma(half16, half16, float8)
    // See: https://codebrowser.dev/llvm/clang/include/clang/Basic/BuiltinsAMDGPU.def.html
    const float8_t r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, c_vec);

    // Unpack result
    #pragma unroll
    for (int i = 0; i < 8; i++) d.x[i] = r[i];
#else
    // Fallback: copy accumulator unchanged
    #pragma unroll
    for (int i = 0; i < 8; i++) d.x[i] = c.x[i];
#endif
}

// ---------------------------
// Accumulator store (tile-local, row-major)
// Correct lane mapping for wave32:
//   lane%16 => column
//   lane/16 => even/odd rows (0 or 1)
//   i=0..7 => rows 0,2,4,6,8,10,12,14 + rg offset
// ---------------------------
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void store_matrix_sync(
    float* out,  // points to C tile base (row-major)
    const fragment<accumulator, M, N, K, float, Layout>& frag,
    int ldm
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int col  = lane & 15;       // column = lane mod 16
    const int rg   = lane >> 4;       // row group: 0 or 1

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int row = (i << 1) + rg; // rows: 0,2,4,6,8,10,12,14 + rg
        out[row * ldm + col] = frag.x[i];
    }
}

// ---------------------------
// Global memory loaders (non-swizzled, for initial load)
// ---------------------------
template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, M, N, K, __half, Layout>& frag,
    const __half* ptr,
    int ldm
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int row = lane & 15;
    const __half* row_ptr = ptr + row * ldm;
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        frag.x[i] = row_ptr[i];
    }
}

template<int M, int N, int K, typename Layout>
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, M, N, K, __half, Layout>& frag,
    const __half* ptr,
    int ldm
) {
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const int col = lane & 15;
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        frag.x[i] = ptr[i * ldm + col];
    }
}

} // namespace rocwmma

#endif // ROCWMMA_GFX1151_PATCH_HPP
