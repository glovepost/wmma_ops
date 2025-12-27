#!/usr/bin/env python3
"""Final performance summary for WMMA kernel implementation."""
import torch
import wmma_ops

print("Final Performance Summary - Updated Adaptive Selector")
print("="*80)

# Test typical ML matrix sizes
test_cases = [
    (256, 256, 256, "Tiny"),
    (512, 512, 512, "Small"),
    (768, 768, 768, "Medium-Small"),
    (1024, 1024, 1024, "Medium"),
    (1536, 1536, 1536, "Medium-Large"),
    (2048, 2048, 2048, "Large"),
    (4096, 4096, 4096, "XLarge"),
    (4096, 11008, 4096, "LLaMA MLP")
]

print(f"{'Shape':25s}  {'TFLOPS':>8s}  {'vs Peak':>8s}  {'% rocBLAS':>10s}")
print("-"*80)

total_adaptive = 0
total_rocblas = 0
count = 0

for M, N, K, label in test_cases:
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    
    # Warmup
    for _ in range(3):
        _ = wmma_ops.matmul_adaptive(A, B)
        _ = torch.mm(A, B)  # FP16 matmul
    torch.cuda.synchronize()
    
    # Benchmark adaptive
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = 20
    
    start.record()
    for _ in range(iters):
        _ = wmma_ops.matmul_adaptive(A, B)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    ops = 2.0 * M * N * K
    tflops_adaptive = ops / (time_ms * 1e-3) / 1e12
    
    # Benchmark rocBLAS FP16 (correct comparison)
    start.record()
    for _ in range(iters):
        _ = torch.mm(A, B)  # FP16 matmul, not FP32!
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / iters
    tflops_rocblas = ops / (time_ms * 1e-3) / 1e12
    
    peak_pct = 100 * tflops_adaptive / 59.4
    rocblas_pct = 100 * tflops_adaptive / tflops_rocblas
    
    shape = f"{M}x{N}x{K} ({label})"
    print(f"{shape:25s}  {tflops_adaptive:8.2f}  {peak_pct:6.1f}%   {rocblas_pct:8.1f}%")
    
    total_adaptive += tflops_adaptive
    total_rocblas += tflops_rocblas
    count += 1

print("-"*80)
avg_adaptive = total_adaptive / count
avg_rocblas = total_rocblas / count
avg_peak = 100 * avg_adaptive / 59.4
avg_rocblas_pct = 100 * avg_adaptive / avg_rocblas
print(f"{'AVERAGE':25s}  {avg_adaptive:8.2f}  {avg_peak:6.1f}%   {avg_rocblas_pct:8.1f}%")
print()
print(f"Peak FP16 WMMA: 59.4 TFLOPS (theoretical gfx1151)")
print(f"rocBLAS FP16 avg: {avg_rocblas:.1f} TFLOPS ({100*avg_rocblas/59.4:.0f}% of peak)")
print(f"Our WMMA avg: {avg_adaptive:.1f} TFLOPS ({avg_peak:.0f}% of peak, {avg_rocblas_pct:.0f}% of rocBLAS)")

