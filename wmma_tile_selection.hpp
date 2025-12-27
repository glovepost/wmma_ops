// ============================================================================
// WMMA TILE SELECTION LOGIC
// For AMD gfx1151 (RDNA3.5 / Strix Halo)
// 
// Adaptive tile configuration selection based on matrix dimensions
// ============================================================================

#ifndef WMMA_TILE_SELECTION_HPP
#define WMMA_TILE_SELECTION_HPP

#include <algorithm>

// ============================================================================
// TILE CONFIGURATION ENUM
// ============================================================================

enum class TileConfig {
    SMALL_64x64 = 0,    // Best for M,N < 512
    MEDIUM_128x64 = 1,  // Best for most cases
    LARGE_256x64 = 2,   // Best for M > 2048
    WIDE_128x128 = 3,   // Best for square matrices with N > 1024
    K_UNROLL = 4        // Best for 768-1536 range (reduced sync overhead)
};

// ============================================================================
// ADAPTIVE TILE SELECTION
// Selects optimal kernel based on matrix dimensions for gfx1151
// ============================================================================

__host__ inline TileConfig select_optimal_tile(int M, int N, int K) {
    // Heuristics tuned via Optuna TPE sampler on gfx1151
    // Based on auto-tuning 13 common ML matrix sizes
    // See autotune.py for tuning methodology
    
    const int max_dim = std::max(M, N);
    const float aspect = static_cast<float>(M) / static_cast<float>(N);
    
    // Very small matrices (< 512): 64x64 for better occupancy
    // Tuning: 256x256x256 -> 3.4 TFLOPS with 64x64
    if (max_dim < 512) {
        return TileConfig::SMALL_64x64;
    }
    
    // Tall matrices (M >> N): 64x64 for better thread utilization
    // Tuning: 8192x1024x2048 -> 13.7 TFLOPS with 64x64
    if (aspect > 4.0f && M >= 4096) {
        return TileConfig::SMALL_64x64;
    }
    
    // LLaMA MLP shapes (wide N): 256x64 for better B reuse
    // Tuning: 4096x11008x4096 -> 21.1 TFLOPS with 256x64
    if (N >= 8192 && M >= 2048) {
        return TileConfig::LARGE_256x64;
    }
    
    // Medium matrices (768-1536): K-unroll wins by 30%
    // Tuning: 1024x1024x1024 -> 13.76 TFLOPS with K-Unroll vs 10.37 for standard
    // K-unroll reduces __syncthreads overhead by processing 2 K-tiles per sync
    if (max_dim >= 768 && max_dim <= 1536 && K >= 512) {
        return TileConfig::K_UNROLL;
    }
    
    // Default: 128x64 tile wins for 70% of tested sizes
    // Peak: 21.6 TFLOPS at 4096x4096x4096
    return TileConfig::MEDIUM_128x64;
}

#endif // WMMA_TILE_SELECTION_HPP

