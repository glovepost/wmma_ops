#!/usr/bin/env python3
"""
Fragment Loading Correctness Test
==================================

Tests the asmOpt kernel's fragment loading logic against PyTorch reference
to verify correctness on real hardware.

This test specifically focuses on fragment loading patterns to diagnose
issues identified in the fragment layout research.

Usage:
    # Ensure environment is set up (PyTorch with ROCm support)
    cd extensions/wmma_ops
    python3 test_fragment_loading.py

    # Or via build script:
    ./build_and_test.sh
    python3 test_fragment_loading.py
"""

import sys
import os
import traceback

# Set up environment (matching test_rocwmma_patch.py)
tunableop_dir = "/tmp/tunableop"
os.makedirs(tunableop_dir, mode=0o777, exist_ok=True)
try:
    os.chmod(tunableop_dir, 0o777)
except OSError:
    pass

if "TUNABLEOP_RESULTS_DIR" not in os.environ:
    os.environ["TUNABLEOP_RESULTS_DIR"] = tunableop_dir

try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{current_ld_path}"
except ImportError:
    pass

import torch
import numpy as np

def test_fragment_loading_correctness():
    """Test asmOpt kernel correctness with detailed error analysis"""
    try:
        import wmma_ops
        print("✅ wmma_ops module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import wmma_ops: {e}")
        print("   Build with: cd extensions/wmma_ops && pip install -e .")
        return False
    
    print("\n" + "="*80)
    print("FRAGMENT LOADING CORRECTNESS TEST")
    print("="*80)
    print("\nTesting asmOpt kernel fragment loading against PyTorch reference...")
    print("Kernel: wmma_gemm_kernel_asmOpt (uses padded LDS, no XOR swizzle)\n")
    
    # Test cases: various matrix sizes to catch different access patterns
    test_cases = [
        (512, 512, 64, "Small square (K=64)"),
        (512, 512, 128, "Small square (K=128)"),
        (1024, 1024, 256, "Medium square (K=256)"),
        (2048, 2048, 512, "Large square (K=512)"),
        (4096, 4096, 1024, "XL square (K=1024)"),
        (2048, 1024, 512, "Rectangular M>N"),
        (1024, 2048, 512, "Rectangular N>M"),
    ]
    
    results = []
    
    for M, N, K, description in test_cases:
        print(f"\n{'─'*80}")
        print(f"Test: {description} (M={M}, N={N}, K={K})")
        print(f"{'─'*80}")
        
        # Generate test matrices with known pattern for easier debugging
        # Use deterministic values for reproducibility
        torch.manual_seed(42)
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Reference computation (FP32 for accuracy)
        with torch.cuda.device(A.device):
            torch.cuda.synchronize()
            C_ref = torch.matmul(A.float(), B.float())
        
        # Test asmOpt kernel
        try:
            with torch.cuda.device(A.device):
                torch.cuda.synchronize()
                C_asmOpt = wmma_ops.matmul_asmOpt(A, B)
                torch.cuda.synchronize()
        except Exception as e:
            print(f"  ❌ Kernel execution failed: {e}")
            traceback.print_exc()
            results.append((description, False, 0, 0, 0, str(e)))
            continue
        
        # Convert to float32 for comparison
        C_asmOpt_f32 = C_asmOpt.float()
        
        # Compute errors
        abs_error = (C_asmOpt_f32 - C_ref).abs()
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        
        # Relative error (avoid division by zero)
        ref_abs_max = C_ref.abs().max().item()
        if ref_abs_max > 0:
            max_rel_error = max_abs_error / ref_abs_max
        else:
            max_rel_error = float('inf') if max_abs_error > 0 else 0.0
        
        # L2 norm of error
        error_norm = torch.norm(C_asmOpt_f32 - C_ref).item()
        ref_norm = torch.norm(C_ref).item()
        rel_l2_error = error_norm / ref_norm if ref_norm > 0 else float('inf')
        
        # Determine if test passed
        # For FP16 WMMA, we expect some error due to precision, but it should be small
        # Typical good kernels have max_rel_error < 0.1% or max_abs_error < 0.1
        passed = (max_rel_error < 0.01) or (max_abs_error < 1.0)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"  Status: {status}")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Max relative error:  {max_rel_error*100:.4f}%")
        print(f"  Relative L2 error:   {rel_l2_error*100:.4f}%")
        
        # Additional diagnostics for failed tests
        if not passed:
            print(f"\n  ⚠️  Correctness issue detected!")
            print(f"     Error statistics:")
            print(f"       - Error range: [{abs_error.min().item():.6f}, {abs_error.max().item():.6f}]")
            print(f"       - Error std:   {abs_error.std().item():.6f}")
            
            # Check if errors are systematic or random
            error_fraction_large = (abs_error > 1.0).float().mean().item()
            if error_fraction_large > 0.1:
                print(f"       - {error_fraction_large*100:.1f}% of elements have error > 1.0 (systematic issue)")
            else:
                print(f"       - {error_fraction_large*100:.1f}% of elements have error > 1.0 (sparse errors)")
        
        results.append((description, passed, max_abs_error, max_rel_error, rel_l2_error, None))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, p, _, _, _, _ in results if p)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count} tests\n")
    
    for desc, passed, max_abs, max_rel, rel_l2, error in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {desc:30s}  max_err={max_abs:.4f}  rel_err={max_rel*100:.2f}%")
        if error:
            print(f"     Error: {error}")
    
    return passed_count == total_count


def compare_with_working_kernel():
    """Compare asmOpt with a known-working kernel (standard matmul)"""
    try:
        import wmma_ops
    except ImportError:
        print("❌ wmma_ops not available")
        return
    
    print("\n" + "="*80)
    print("COMPARISON: asmOpt vs Standard Kernel")
    print("="*80)
    
    # Test matrix
    M, N, K = 2048, 2048, 512
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Reference
    C_ref = torch.matmul(A.float(), B.float())
    
    # Standard kernel (known to work)
    C_standard = wmma_ops.matmul(A, B).float()
    
    # asmOpt kernel
    C_asmOpt = wmma_ops.matmul_asmOpt(A, B).float()
    
    # Compare errors
    error_standard = (C_standard - C_ref).abs().max().item()
    error_asmOpt = (C_asmOpt - C_ref).abs().max().item()
    
    print(f"\nStandard kernel max error: {error_standard:.6f}")
    print(f"asmOpt kernel max error:    {error_asmOpt:.6f}")
    
    if error_asmOpt > error_standard * 10:
        print(f"\n⚠️  asmOpt error is {error_asmOpt/error_standard:.1f}x larger than standard kernel")
        print("   This suggests a correctness issue in asmOpt")
    else:
        print(f"\n✅ asmOpt error is comparable to standard kernel")
    
    # Direct comparison
    diff_kernels = (C_asmOpt - C_standard).abs()
    max_diff = diff_kernels.max().item()
    mean_diff = diff_kernels.mean().item()
    
    print(f"\nDirect kernel difference:")
    print(f"  Max difference:  {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")


def test_small_cases():
    """Test very small matrices where errors are easier to analyze"""
    try:
        import wmma_ops
    except ImportError:
        return
    
    print("\n" + "="*80)
    print("DETAILED SMALL CASE ANALYSIS")
    print("="*80)
    
    # Very small test case (easy to debug)
    M, N, K = 64, 64, 16
    print(f"\nTest: M={M}, N={N}, K={K} (smallest meaningful size)")
    
    # Use simple pattern: A[i][j] = i, B[j][k] = j
    A = torch.zeros(M, K, device='cuda', dtype=torch.float16)
    B = torch.zeros(K, N, device='cuda', dtype=torch.float16)
    
    for i in range(M):
        for j in range(K):
            A[i, j] = float(i % 16)  # Pattern repeats every 16
    
    for j in range(K):
        for k in range(N):
            B[j, k] = float(j)
    
    # Reference: C[i][k] = sum_j(A[i][j] * B[j][k])
    C_ref = torch.matmul(A.float(), B.float())
    
    # Test asmOpt
    C_asmOpt = wmma_ops.matmul_asmOpt(A, B).float()
    
    # Detailed error analysis
    error = (C_asmOpt - C_ref).abs()
    max_error = error.max().item()
    
    print(f"\nMax error: {max_error:.6f}")
    
    if max_error > 0.01:
        print("\n⚠️  Significant errors detected!")
        print("\nFirst 8x8 block of results:")
        print("Reference:")
        print(C_ref[:8, :8].cpu().numpy())
        print("\nasmOpt:")
        print(C_asmOpt[:8, :8].cpu().numpy())
        print("\nError:")
        print(error[:8, :8].cpu().numpy())
    else:
        print("✅ Small case passed")


if __name__ == "__main__":
    print("Fragment Loading Correctness Test for asmOpt Kernel")
    print("="*80)
    
    # Run tests
    success = test_fragment_loading_correctness()
    compare_with_working_kernel()
    test_small_cases()
    
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - Fragment loading may need correction")
    print("="*80)
    
    sys.exit(0 if success else 1)

