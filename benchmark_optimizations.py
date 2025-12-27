#!/usr/bin/env python3
"""
Benchmark script for WMMA kernel optimizations
Compares different kernel variants including XOR swizzle optimizations
"""
import torch
import time
import sys
import os

# Setup environment
tunableop_dir = "/tmp/tunableop"
os.makedirs(tunableop_dir, mode=0o777, exist_ok=True)
os.environ.setdefault("PYTORCH_TUNABLEOP_RESULT_DIR", tunableop_dir)

try:
    import wmma_ops
except ImportError as e:
    print(f"❌ Failed to import wmma_ops: {e}")
    print("   Make sure the extension is built: pip install -e .")
    sys.exit(1)

def benchmark_kernel(kernel_func, A, B, iters=20, warmup=3):
    """Benchmark a kernel function and return TFLOPS and time_ms"""
    M, K = A.shape
    _, N = B.shape
    
    # Warmup
    for _ in range(warmup):
        _ = kernel_func(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        result = kernel_func(A, B)
    end_event.record()
    torch.cuda.synchronize()
    
    time_ms = start_event.elapsed_time(end_event) / iters
    ops = 2.0 * M * N * K
    tflops = ops / (time_ms * 1e-3) / 1e12
    
    return tflops, time_ms, result

def test_correctness(kernel_func, A, B, ref_result, name):
    """Test kernel correctness against reference"""
    try:
        result = kernel_func(A, B)
        max_err = (result - ref_result).abs().max().item()
        ref_max = ref_result.abs().max().item()
        rel_err = max_err / ref_max if ref_max > 0 else float('inf')
        
        if rel_err < 0.01:  # 1% relative error threshold
            return True, rel_err
        else:
            return False, rel_err
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("="*80)
    print("WMMA Kernel Optimization Benchmark")
    print("="*80)
    print()
    
    # Test configurations
    test_sizes = [
        (512, 512, 512, "Small"),
        (1024, 1024, 1024, "Medium"),
        (2048, 2048, 2048, "Large"),
        (4096, 4096, 4096, "XLarge"),
    ]
    
    # Kernel variants to test
    # Note: Swizzled and ASM-Opt kernels have known correctness issues
    kernels = [
        ("Standard (matmul)", wmma_ops.matmul),
        ("Adaptive", wmma_ops.matmul_adaptive),
        ("K-Unroll", wmma_ops.matmul_kunroll),
        ("HighOcc", wmma_ops.matmul_highOcc),
        ("NoPrefetch", wmma_ops.matmul_noPrefetch),
    ]
    
    # Optional: Test kernels with known issues (will show as FAIL)
    test_broken = False
    if test_broken:
        kernels.extend([
            ("Swizzled (XOR)", wmma_ops.matmul_swizzled),  # Known correctness issues
            ("ASM-Opt", wmma_ops.matmul_asmOpt),  # Known correctness issues
        ])
    
    # Results storage
    results = {}
    for name, _ in kernels:
        results[name] = {}
    
    print(f"{'Size':20s} {'Kernel':20s} {'TFLOPS':>10s} {'Time(ms)':>10s} {'Status':>10s}")
    print("-"*80)
    
    for M, N, K, label in test_sizes:
        print(f"\n{label} ({M}x{N}x{K}):")
        
        # Create test matrices
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        
        # Reference result (PyTorch FP32 for accuracy)
        ref = torch.matmul(A.float(), B.float())
        
        # Benchmark each kernel
        for name, kernel_func in kernels:
            try:
                # Benchmark
                tflops, time_ms, result = benchmark_kernel(kernel_func, A, B)
                
                # Test correctness
                correct, error = test_correctness(kernel_func, A, B, ref, name)
                status = "✅ OK" if correct else f"❌ ERR({error:.4f})"
                
                results[name][label] = {
                    'tflops': tflops,
                    'time_ms': time_ms,
                    'correct': correct,
                    'error': error
                }
                
                print(f"  {name:20s} {tflops:10.2f} {time_ms:10.3f} {status:>10s}")
                
            except Exception as e:
                print(f"  {name:20s} {'FAILED':>10s} {'-':>10s} {str(e)[:20]}")
                results[name][label] = {'tflops': 0, 'time_ms': 0, 'correct': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("Performance Summary (TFLOPS)")
    print("="*80)
    print(f"{'Kernel':20s}", end="")
    for _, _, _, label in test_sizes:
        print(f" {label:>10s}", end="")
    print("  Average")
    print("-"*80)
    
    for name, _ in kernels:
        print(f"{name:20s}", end="")
        tflops_list = []
        for _, _, _, label in test_sizes:
            if label in results[name]:
                t = results[name][label]['tflops']
                print(f" {t:10.2f}", end="")
                tflops_list.append(t)
            else:
                print(f" {'-':>10s}", end="")
        if tflops_list:
            avg = sum(tflops_list) / len(tflops_list)
            print(f"  {avg:10.2f}")
        else:
            print()
    
    # Correctness summary
    print("\n" + "="*80)
    print("Correctness Summary")
    print("="*80)
    for name, _ in kernels:
        all_correct = all(
            results[name].get(label, {}).get('correct', False)
            for _, _, _, label in test_sizes
        )
        status = "✅ PASS" if all_correct else "❌ FAIL"
        print(f"{name:30s} {status}")
    
    # Best kernel per size
    print("\n" + "="*80)
    print("Best Kernel by Size")
    print("="*80)
    for _, _, _, label in test_sizes:
        best_name = ""
        best_tflops = 0
        for name, _ in kernels:
            if label in results[name] and results[name][label]['correct']:
                t = results[name][label]['tflops']
                if t > best_tflops:
                    best_tflops = t
                    best_name = name
        if best_name:
            print(f"{label:20s} {best_name:30s} {best_tflops:8.2f} TFLOPS")
    
    print("\n" + "="*80)
    print("Peak theoretical: 59.4 TFLOPS (gfx1151 FP16 WMMA)")
    print("="*80)

if __name__ == "__main__":
    main()

