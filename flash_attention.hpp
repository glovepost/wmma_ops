// ============================================================================
// FLASH ATTENTION V2 FOR AMD gfx1151 (RDNA3.5 / Strix Halo)
// 
// Based on: https://github.com/Repeerc/flash-attention-v2-RDNA3-minimal
// Simplified implementation for correctness first, then optimize
//
// Flash Attention Algorithm (Dao et al.):
// - Tiled computation to fit in SRAM (LDS)
// - Online softmax with running max/sum
// - O(N) memory instead of O(N²)
// ============================================================================

#ifndef FLASH_ATTENTION_HPP
#define FLASH_ATTENTION_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr int FA_WAVE_SIZE = 32;
constexpr float FA_MAX_NUM = 30000.0f;  // Safe max for FP16

// ============================================================================
// SIMPLE FLASH ATTENTION FORWARD KERNEL
// One thread per query position, simple but correct
// ============================================================================

template<bool CAUSAL>
__launch_bounds__(256)
__global__ void flash_attention_fwd_kernel_simple(
    const __half* __restrict__ Q,  // [B, H, N_q, D]
    const __half* __restrict__ K,  // [B, H, N_kv, D]
    const __half* __restrict__ V,  // [B, H, N_kv, D]
    __half* __restrict__ O,        // [B, H, N_q, D]
    const int N_q,                 // Query sequence length
    const int N_kv,                // Key/Value sequence length
    const int D,                   // Head dimension
    const int64_t stride_qb,       // Q batch stride
    const int64_t stride_qh,       // Q head stride
    const int64_t stride_qn,       // Q sequence stride (= D for contiguous)
    const int64_t stride_kb,
    const int64_t stride_kh,
    const int64_t stride_kn,
    const float scale              // 1/sqrt(D)
) {
    // Grid: (B, H, ceil(N_q / blockDim.x))
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (q_idx >= N_q) return;
    
    // Compute base offsets
    const int64_t q_base = batch_idx * stride_qb + head_idx * stride_qh + q_idx * stride_qn;
    const int64_t kv_base = batch_idx * stride_kb + head_idx * stride_kh;
    const int64_t o_base = batch_idx * stride_qb + head_idx * stride_qh + q_idx * stride_qn;
    
    // Load Q vector for this position
    float q_vec[128];  // Max D = 128
    for (int d = 0; d < D; d++) {
        q_vec[d] = __half2float(Q[q_base + d]);
    }
    
    // Online softmax variables
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum of exp
    float o_vec[128] = {0}; // Output accumulator
    
    // Loop over all K/V positions
    const int kv_end = CAUSAL ? min(N_kv, q_idx + 1) : N_kv;
    
    for (int kv_idx = 0; kv_idx < kv_end; kv_idx++) {
        const int64_t k_offset = kv_base + kv_idx * stride_kn;
        const int64_t v_offset = kv_base + kv_idx * stride_kn;
        
        // Compute attention score: Q[q_idx] · K[kv_idx]
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            score += q_vec[d] * __half2float(K[k_offset + d]);
        }
        score *= scale;
        
        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float exp_diff = expf(m_i - m_new);  // exp(m_i - m_new)
        float exp_score = expf(score - m_new);  // exp(score - m_new)
        
        // Update running sum
        l_i = l_i * exp_diff + exp_score;
        
        // Update output: O = O * exp_diff + exp_score * V
        for (int d = 0; d < D; d++) {
            float v_val = __half2float(V[v_offset + d]);
            o_vec[d] = o_vec[d] * exp_diff + exp_score * v_val;
        }
        
        m_i = m_new;
    }
    
    // Final normalization: O = O / l_i
    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = 0; d < D; d++) {
        O[o_base + d] = __float2half(o_vec[d] * inv_l);
    }
}

// Template instantiations
template __global__ void flash_attention_fwd_kernel_simple<false>(
    const __half*, const __half*, const __half*, __half*,
    int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

template __global__ void flash_attention_fwd_kernel_simple<true>(
    const __half*, const __half*, const __half*, __half*,
    int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float);

#endif // FLASH_ATTENTION_HPP
