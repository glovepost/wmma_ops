// ============================================================================
// WMMA DEVICE HELPER FUNCTIONS
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
// 
// Architecture-specific device helper functions for WMMA kernels
// ============================================================================

#ifndef WMMA_DEVICE_HELPERS_HPP
#define WMMA_DEVICE_HELPERS_HPP

#include "wmma_xor_swizzle.hpp"  // For type definitions (half8, etc.)

// ============================================================================
// GFX1151-SPECIFIC INTRINSICS AND ASSEMBLY HELPERS
// These are architecture-specific and NOT portable
// ============================================================================

// Direct vectorized load to LDS via registers (ROCm 6.x compatible)
// RDNA3 can dual-issue global_load + WMMA, so this still benefits from interleaving
__device__ __forceinline__ void load_half8_to_lds(
    const __half* __restrict__ global_ptr,
    __half* lds_ptr,
    bool is_valid
) {
    if (is_valid) {
        half8 data = *reinterpret_cast<const half8*>(global_ptr);
        *reinterpret_cast<half8*>(lds_ptr) = data;
    }
}

// Hardware data prefetch for global memory
// Note: s_prefetch_data is not available on RDNA3.5, use software prefetch
__device__ __forceinline__ void hw_prefetch_global(const void* ptr, int size_bytes) {
    // Use compiler builtin for prefetch
    __builtin_prefetch(ptr, 0, 3);
}

// Explicit memory fence for LDS
__device__ __forceinline__ void lds_fence() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// Explicit memory fence for VMEM
__device__ __forceinline__ void vmem_fence() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// Combined fence
__device__ __forceinline__ void full_fence() {
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

#endif // WMMA_DEVICE_HELPERS_HPP

