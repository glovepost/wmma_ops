#!/usr/bin/env python3
"""
Test script for rocWMMA patch optimizations
Tests correctness and benchmarks performance
"""

import sys
import os
import time
import traceback
import warnings

# Create tunableop directory and ensure it's writable
# PyTorch's tunable ops system needs this directory to save tuning results
tunableop_dir = "/tmp/tunableop"
os.makedirs(tunableop_dir, mode=0o777, exist_ok=True)

# Ensure the directory is writable by setting proper permissions
try:
    os.chmod(tunableop_dir, 0o777)
except OSError:
    pass  # Ignore if we can't change permissions

# Set environment variable to point to the tunableop directory
# This helps PyTorch find the correct location
if "TUNABLEOP_RESULTS_DIR" not in os.environ:
    os.environ["TUNABLEOP_RESULTS_DIR"] = tunableop_dir

# Set library path for PyTorch extensions
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{current_ld_path}"
except ImportError:
    pass

def test_import():
    """Test if the module can be imported"""
    try:
        import wmma_ops
        print("✅ wmma_ops module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import wmma_ops: {e}")
        return False

def test_correctness():
    """Test correctness against PyTorch reference"""
    try:
        import torch
        import wmma_ops
        
        print("\n📊 Testing correctness...")
        
        # Test case 1: Small matrix (K <= 96) - should use 8 warps
        print("  Test 1: Small matrix (M=512, N=512, K=64)")
        A1 = torch.randn(512, 64, device='cuda', dtype=torch.float16)
        B1 = torch.randn(64, 512, device='cuda', dtype=torch.float16)
        
        C1_wmma = wmma_ops.matmul(A1, B1)
        C1_ref = torch.matmul(A1, B1)
        
        max_err1 = (C1_wmma - C1_ref).abs().max().item()
        rel_err1 = max_err1 / C1_ref.abs().max().item()
        
        print(f"    Max absolute error: {max_err1:.6f}")
        print(f"    Max relative error: {rel_err1:.6f}")
        
        # For FP16 WMMA, relative error < 0.1% is acceptable
        # Absolute error threshold depends on matrix size
        if rel_err1 < 0.001 or max_err1 < 0.1:
            print("    ✅ Test 1 passed")
        else:
            print(f"    ❌ Test 1 failed (error too large)")
            return False
        
        # Test case 2: Medium matrix (K > 96) - should use 4 warps
        print("\n  Test 2: Medium matrix (M=2048, N=2048, K=128)")
        A2 = torch.randn(2048, 128, device='cuda', dtype=torch.float16)
        B2 = torch.randn(128, 2048, device='cuda', dtype=torch.float16)
        
        C2_wmma = wmma_ops.matmul(A2, B2)
        C2_ref = torch.matmul(A2, B2)
        
        max_err2 = (C2_wmma - C2_ref).abs().max().item()
        rel_err2 = max_err2 / C2_ref.abs().max().item()
        
        print(f"    Max absolute error: {max_err2:.6f}")
        print(f"    Max relative error: {rel_err2:.6f}")
        
        # For FP16 WMMA, relative error < 0.1% is acceptable
        if rel_err2 < 0.001 or max_err2 < 0.1:
            print("    ✅ Test 2 passed")
        else:
            print(f"    ❌ Test 2 failed (error too large)")
            return False
        
        # Test case 3: Large matrix
        print("\n  Test 3: Large matrix (M=4096, N=4096, K=2048)")
        A3 = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)
        B3 = torch.randn(2048, 4096, device='cuda', dtype=torch.float16)
        
        C3_wmma = wmma_ops.matmul(A3, B3)
        C3_ref = torch.matmul(A3, B3)
        
        max_err3 = (C3_wmma - C3_ref).abs().max().item()
        rel_err3 = max_err3 / C3_ref.abs().max().item()
        
        print(f"    Max absolute error: {max_err3:.6f}")
        print(f"    Max relative error: {rel_err3:.6f}")
        
        # For FP16 WMMA on large matrices, relative error < 0.1% is acceptable
        if rel_err3 < 0.001 or max_err3 < 0.5:
            print("    ✅ Test 3 passed")
        else:
            print(f"    ❌ Test 3 failed (error too large)")
            return False
        
        print("\n✅ All correctness tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Correctness test failed: {e}")
        traceback.print_exc()
        return False

def benchmark_performance():
    """Benchmark performance and compare with PyTorch"""
    try:
        import torch
        import wmma_ops
        
        print("\n⚡ Benchmarking performance...")
        
        def benchmark(name, func, *args, iterations=100, warmup=10):
            # Warmup
            for _ in range(warmup):
                _ = func(*args)
            torch.cuda.synchronize()
            
            # Time
            start = time.perf_counter()
            for _ in range(iterations):
                result = func(*args)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            return elapsed / iterations, result
        
        # Benchmark 1: Small K (should use 8 warps)
        print("\n  Benchmark 1: Small K (M=2048, N=2048, K=64)")
        A1 = torch.randn(2048, 64, device='cuda', dtype=torch.float16)
        B1 = torch.randn(64, 2048, device='cuda', dtype=torch.float16)
        
        t_wmma1, _ = benchmark("WMMA", wmma_ops.matmul, A1, B1)
        t_ref1, _ = benchmark("PyTorch", torch.matmul, A1, B1)
        
        flops1 = 2 * 2048 * 64 * 2048
        tflops_wmma1 = flops1 / t_wmma1 / 1e12
        tflops_ref1 = flops1 / t_ref1 / 1e12
        
        print(f"    WMMA: {t_wmma1*1000:.3f} ms ({tflops_wmma1:.2f} TFLOPS)")
        print(f"    PyTorch: {t_ref1*1000:.3f} ms ({tflops_ref1:.2f} TFLOPS)")
        speedup1 = t_ref1/t_wmma1
        print(f"    Speedup: {speedup1:.2f}x ({'slower' if speedup1 < 1.0 else 'faster'})")
        if speedup1 < 0.5:
            print(f"    ⚠️  Performance issue: Memory-bound (needs shared memory caching)")
        
        # Benchmark 2: Medium K (should use 4 warps)
        print("\n  Benchmark 2: Medium K (M=4096, N=4096, K=128)")
        A2 = torch.randn(4096, 128, device='cuda', dtype=torch.float16)
        B2 = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
        
        t_wmma2, _ = benchmark("WMMA", wmma_ops.matmul, A2, B2)
        t_ref2, _ = benchmark("PyTorch", torch.matmul, A2, B2)
        
        flops2 = 2 * 4096 * 128 * 4096
        tflops_wmma2 = flops2 / t_wmma2 / 1e12
        tflops_ref2 = flops2 / t_ref2 / 1e12
        
        print(f"    WMMA: {t_wmma2*1000:.3f} ms ({tflops_wmma2:.2f} TFLOPS)")
        print(f"    PyTorch: {t_ref2*1000:.3f} ms ({tflops_ref2:.2f} TFLOPS)")
        speedup2 = t_ref2/t_wmma2
        print(f"    Speedup: {speedup2:.2f}x ({'slower' if speedup2 < 1.0 else 'faster'})")
        if speedup2 < 0.5:
            print(f"    ⚠️  Performance issue: Memory-bound (needs shared memory caching)")
        
        # Benchmark 3: Large K (should use 4 warps)
        print("\n  Benchmark 3: Large K (M=4096, N=4096, K=2048)")
        A3 = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)
        B3 = torch.randn(2048, 4096, device='cuda', dtype=torch.float16)
        
        t_wmma3, _ = benchmark("WMMA", wmma_ops.matmul, A3, B3, iterations=50)
        t_ref3, _ = benchmark("PyTorch", torch.matmul, A3, B3, iterations=50)
        
        flops3 = 2 * 4096 * 2048 * 4096
        tflops_wmma3 = flops3 / t_wmma3 / 1e12
        tflops_ref3 = flops3 / t_ref3 / 1e12
        
        print(f"    WMMA: {t_wmma3*1000:.3f} ms ({tflops_wmma3:.2f} TFLOPS)")
        print(f"    PyTorch: {t_ref3*1000:.3f} ms ({tflops_ref3:.2f} TFLOPS)")
        speedup3 = t_ref3/t_wmma3
        print(f"    Speedup: {speedup3:.2f}x ({'slower' if speedup3 < 1.0 else 'faster'})")
        if speedup3 < 0.5:
            print(f"    ⚠️  Performance issue: Memory-bound (needs shared memory caching)")
        
        # Summary analysis
        print("\n  📊 Performance Analysis:")
        avg_tflops = (tflops_wmma1 + tflops_wmma2 + tflops_wmma3) / 3
        peak_tflops = 59.4  # gfx1151 WMMA peak
        utilization = (avg_tflops / peak_tflops) * 100
        print(f"    Average WMMA performance: {avg_tflops:.2f} TFLOPS")
        print(f"    Peak WMMA (gfx1151): {peak_tflops:.1f} TFLOPS")
        print(f"    Compute utilization: {utilization:.1f}%")
        if utilization < 10:
            print(f"    ⚠️  Low utilization: Kernel is memory-bound, not compute-bound")
            print(f"    💡 Recommendation: Add shared memory (LDS) caching")
        
        print("\n✅ Performance benchmarks completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("rocWMMA Patch Test Suite")
    print("=" * 60)
    
    # Test import
    if not test_import():
        print("\n❌ Cannot proceed without wmma_ops module")
        return 1
    
    # Test correctness
    if not test_correctness():
        print("\n❌ Correctness tests failed")
        return 1
    
    # Benchmark performance
    if not benchmark_performance():
        print("\n⚠️  Performance benchmarks failed (non-fatal)")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

