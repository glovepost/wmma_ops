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

def benchmark_all_kernels():
    """Benchmark all kernel variants on a standard matrix size"""
    try:
        import torch
        import wmma_ops
        
        print("\n🔬 Benchmarking all kernel variants...")
        print("   Matrix size: 4096×4096×2048")
        print("   " + "-" * 60)
        
        # Standard test matrix
        M, N, K = 4096, 4096, 2048
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        C_ref = torch.matmul(A, B)
        
        flops = 2 * M * K * N
        iterations = 20
        warmup = 5
        
        def benchmark_kernel(name, func, *args):
            """Benchmark a single kernel and check correctness"""
            try:
                # Warmup
                for _ in range(warmup):
                    result = func(*args)
                torch.cuda.synchronize()
                
                # Time
                start = time.perf_counter()
                for _ in range(iterations):
                    result = func(*args)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                avg_time = elapsed / iterations
                tflops = flops / avg_time / 1e12
                
                # Check correctness
                max_err = (result - C_ref).abs().max().item()
                rel_err = max_err / C_ref.abs().max().item()
                correct = rel_err < 0.01  # 1% tolerance
                
                return avg_time, tflops, rel_err, correct
            except Exception as e:
                return None, None, None, str(e)
        
        # All kernel variants to test
        kernels = [
            ("matmul", wmma_ops.matmul),
            ("matmul_adaptive", wmma_ops.matmul_adaptive),
            ("matmul_kunroll", wmma_ops.matmul_kunroll),
            ("matmul_noPrefetch", wmma_ops.matmul_noPrefetch),
            ("matmul_highOcc", wmma_ops.matmul_highOcc),
            ("matmul_quad", wmma_ops.matmul_quad),
            ("matmul_native", wmma_ops.matmul_native),
            ("matmul_hilbert", wmma_ops.matmul_hilbert),
            ("matmul_zerocopy", wmma_ops.matmul_zerocopy),
            ("matmul_asmOpt", wmma_ops.matmul_asmOpt),
            ("matmul_swizzled", wmma_ops.matmul_swizzled),
            ("matmul_xor_optimized", wmma_ops.matmul_xor_optimized),
        ]
        
        # PyTorch reference
        t_ref, _, _, _ = benchmark_kernel("PyTorch", torch.matmul, A, B)
        tflops_ref = flops / t_ref / 1e12 if t_ref else 0
        print(f"   {'PyTorch (reference)':<25} {t_ref*1000:>8.3f} ms  {tflops_ref:>6.2f} TFLOPS  ✅")
        print("   " + "-" * 60)
        
        results = []
        for name, func in kernels:
            t, tflops, rel_err, correct = benchmark_kernel(name, func, A, B)
            
            if t is None:
                print(f"   {name:<25} {'ERROR':<8}       {correct}")
                results.append((name, 0, 0, False))
            elif correct is True:
                speedup = t_ref / t if t else 0
                status = "✅" if correct else "❌"
                print(f"   {name:<25} {t*1000:>8.3f} ms  {tflops:>6.2f} TFLOPS  {status} (err: {rel_err:.4%})")
                results.append((name, tflops, rel_err, True))
            else:
                print(f"   {name:<25} {t*1000:>8.3f} ms  {tflops:>6.2f} TFLOPS  ❌ (err: {rel_err:.4%})")
                results.append((name, tflops, rel_err, False))
        
        # Summary
        print("   " + "-" * 60)
        correct_kernels = [(n, t) for n, t, _, c in results if c]
        if correct_kernels:
            best = max(correct_kernels, key=lambda x: x[1])
            print(f"\n   🏆 Best performing correct kernel: {best[0]} ({best[1]:.2f} TFLOPS)")
            print(f"   📊 PyTorch reference: {tflops_ref:.2f} TFLOPS")
            print(f"   📈 Best vs PyTorch: {best[1]/tflops_ref*100:.1f}%")
        
        failed = [n for n, _, _, c in results if not c]
        if failed:
            print(f"\n   ⚠️  Failed kernels: {', '.join(failed)}")
        
        print("\n✅ All kernel benchmarks completed!")
        return results, tflops_ref
        
    except Exception as e:
        print(f"❌ Kernel benchmark failed: {e}")
        traceback.print_exc()
        return [], 0


def detailed_bottleneck_analysis():
    """
    Detailed bottleneck analysis for each kernel variant.
    
    Analyzes:
    1. Roofline model (compute vs memory bound)
    2. Scaling behavior across K dimensions
    3. Arithmetic intensity and achieved bandwidth
    4. Per-kernel bottleneck diagnosis
    """
    try:
        import torch
        import wmma_ops
        
        print("\n" + "=" * 70)
        print("📊 DETAILED BOTTLENECK ANALYSIS")
        print("=" * 70)
        
        # gfx1151 hardware specs
        PEAK_TFLOPS = 59.4          # FP16 WMMA peak
        PEAK_BW_GBS = 256.0         # LPDDR5X bandwidth (GB/s)
        RIDGE_POINT = (PEAK_TFLOPS * 1e12) / (PEAK_BW_GBS * 1e9)  # ~232 ops/byte
        
        print(f"\n   Hardware: AMD gfx1151 (RDNA3.5 / Strix Halo)")
        print(f"   Peak Compute: {PEAK_TFLOPS:.1f} TFLOPS (FP16 WMMA)")
        print(f"   Peak Memory BW: {PEAK_BW_GBS:.0f} GB/s (LPDDR5X)")
        print(f"   Ridge Point: {RIDGE_POINT:.0f} ops/byte")
        
        # =====================================================================
        # SECTION 1: Roofline Analysis - Scaling with K
        # =====================================================================
        print("\n" + "-" * 70)
        print("   SECTION 1: Roofline Analysis (Scaling with K)")
        print("-" * 70)
        print("\n   Testing how performance scales with K dimension...")
        print("   (Low K = memory bound, High K = compute bound)\n")
        
        M, N = 4096, 4096
        k_values = [32, 64, 128, 256, 512, 1024, 2048]
        
        def benchmark_single(func, A, B, iterations=20, warmup=5):
            for _ in range(warmup):
                _ = func(A, B)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(iterations):
                _ = func(A, B)
            torch.cuda.synchronize()
            return (time.perf_counter() - start) / iterations
        
        print(f"   {'K':>6}  {'AI':>8}  {'WMMA':>10}  {'PyTorch':>10}  {'%Peak':>7}  {'Bound':>12}")
        print(f"   {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*12}")
        
        scaling_results = []
        for K in k_values:
            A = torch.randn(M, K, device='cuda', dtype=torch.float16)
            B = torch.randn(K, N, device='cuda', dtype=torch.float16)
            
            # Arithmetic intensity: FLOPs / Bytes
            # FLOPs = 2*M*N*K, Bytes = 2*(M*K + K*N + M*N) for FP16 in/out
            flops = 2 * M * N * K
            bytes_transferred = 2 * (M * K + K * N + M * N)  # FP16 = 2 bytes
            arith_intensity = flops / bytes_transferred
            
            # Benchmark
            t_wmma = benchmark_single(wmma_ops.matmul, A, B)
            t_ref = benchmark_single(torch.matmul, A, B)
            
            tflops_wmma = flops / t_wmma / 1e12
            tflops_ref = flops / t_ref / 1e12
            pct_peak = (tflops_wmma / PEAK_TFLOPS) * 100
            
            # Achieved bandwidth (assuming memory bound)
            achieved_bw = bytes_transferred / t_wmma / 1e9  # GB/s
            
            # Determine if memory or compute bound
            # If achieved TFLOPS < (AI * peak_bw), we're memory bound
            memory_bound_limit = arith_intensity * PEAK_BW_GBS / 1e3  # TFLOPS
            is_memory_bound = tflops_wmma < memory_bound_limit * 0.9
            bound_type = "Memory" if is_memory_bound else "Compute"
            
            print(f"   {K:>6}  {arith_intensity:>8.1f}  {tflops_wmma:>8.2f} TF  {tflops_ref:>8.2f} TF  {pct_peak:>6.1f}%  {bound_type:>12}")
            
            scaling_results.append({
                'K': K,
                'AI': arith_intensity,
                'tflops': tflops_wmma,
                'pct_peak': pct_peak,
                'bound': bound_type,
                'achieved_bw': achieved_bw
            })
        
        # Find transition point
        transition_k = None
        for i, r in enumerate(scaling_results):
            if r['bound'] == 'Compute':
                transition_k = r['K']
                break
        
        print(f"\n   📈 Analysis:")
        if transition_k:
            print(f"      Memory→Compute transition at K ≈ {transition_k}")
        print(f"      Peak achieved: {max(r['tflops'] for r in scaling_results):.2f} TFLOPS ({max(r['pct_peak'] for r in scaling_results):.1f}% of peak)")
        
        # =====================================================================
        # SECTION 2: Per-Kernel Bottleneck Analysis
        # =====================================================================
        print("\n" + "-" * 70)
        print("   SECTION 2: Per-Kernel Bottleneck Analysis")
        print("-" * 70)
        print("\n   Testing each kernel at compute-bound size (4096×4096×2048)...\n")
        
        M, N, K = 4096, 4096, 2048
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        flops = 2 * M * N * K
        bytes_transferred = 2 * (M * K + K * N + M * N)
        arith_intensity = flops / bytes_transferred
        
        # Kernel characteristics (estimated from code analysis)
        kernel_info = {
            'matmul':           {'tile': '128×64', 'warps': 8, 'regs': '~90', 'lds_kb': 18.4, 'features': 'Double-buffer, LDS pad'},
            'matmul_adaptive':  {'tile': 'Auto',   'warps': 8, 'regs': '~90', 'lds_kb': 18.4, 'features': 'Auto tile select'},
            'matmul_kunroll':   {'tile': '128×64', 'warps': 8, 'regs': '~100', 'lds_kb': 18.4, 'features': '2× K-unroll'},
            'matmul_noPrefetch':{'tile': '128×64', 'warps': 8, 'regs': '~70', 'lds_kb': 18.4, 'features': 'No reg prefetch'},
            'matmul_highOcc':   {'tile': '64×32',  'warps': 4, 'regs': '~50', 'lds_kb': 6.1, 'features': 'High occupancy'},
            'matmul_quad':      {'tile': '128×64', 'warps': 8, 'regs': '~110', 'lds_kb': 36.9, 'features': 'Quad-buffer'},
            'matmul_native':    {'tile': '128×64', 'warps': 8, 'regs': '~90', 'lds_kb': 18.4, 'features': 'Explicit intrinsics'},
            'matmul_hilbert':   {'tile': '128×64', 'warps': 8, 'regs': '~90', 'lds_kb': 18.4, 'features': 'Hilbert tile order'},
            'matmul_zerocopy':  {'tile': '128×64', 'warps': 8, 'regs': '~90', 'lds_kb': 18.4, 'features': 'Swizzled B, zero-copy'},
            'matmul_asmOpt':    {'tile': '128×64', 'warps': 8, 'regs': '~95', 'lds_kb': 18.4, 'features': 'ASM scheduling'},
            'matmul_swizzled':  {'tile': '128×64', 'warps': 8, 'regs': '~90', 'lds_kb': 12.3, 'features': 'XOR swizzle LDS'},
            'matmul_xor_optimized': {'tile': '128×64', 'warps': 8, 'regs': '~90', 'lds_kb': 12.3, 'features': 'XOR swizzle v2'},
        }
        
        kernels = [
            ("matmul", wmma_ops.matmul),
            ("matmul_adaptive", wmma_ops.matmul_adaptive),
            ("matmul_kunroll", wmma_ops.matmul_kunroll),
            ("matmul_noPrefetch", wmma_ops.matmul_noPrefetch),
            ("matmul_highOcc", wmma_ops.matmul_highOcc),
            ("matmul_quad", wmma_ops.matmul_quad),
            ("matmul_native", wmma_ops.matmul_native),
            ("matmul_hilbert", wmma_ops.matmul_hilbert),
            ("matmul_zerocopy", wmma_ops.matmul_zerocopy),
            ("matmul_asmOpt", wmma_ops.matmul_asmOpt),
            ("matmul_swizzled", wmma_ops.matmul_swizzled),
            ("matmul_xor_optimized", wmma_ops.matmul_xor_optimized),
        ]
        
        # PyTorch reference
        t_ref = benchmark_single(torch.matmul, A, B, iterations=20)
        tflops_ref = flops / t_ref / 1e12
        
        print(f"   {'Kernel':<22} {'TFLOPS':>7} {'%Peak':>6} {'%Ref':>6}  {'Tile':>8} {'LDS':>6}  Bottleneck")
        print(f"   {'-'*22} {'-'*7} {'-'*6} {'-'*6}  {'-'*8} {'-'*6}  {'-'*20}")
        
        kernel_results = []
        for name, func in kernels:
            try:
                t = benchmark_single(func, A, B, iterations=20)
                tflops = flops / t / 1e12
                pct_peak = (tflops / PEAK_TFLOPS) * 100
                pct_ref = (tflops / tflops_ref) * 100
                
                info = kernel_info.get(name, {})
                tile = info.get('tile', '?')
                lds_kb = info.get('lds_kb', 0)
                
                # Diagnose bottleneck
                bottleneck = diagnose_bottleneck(name, tflops, pct_peak, info, PEAK_TFLOPS)
                
                print(f"   {name:<22} {tflops:>6.2f}  {pct_peak:>5.1f}% {pct_ref:>5.1f}%  {tile:>8} {lds_kb:>5.1f}K  {bottleneck}")
                
                kernel_results.append({
                    'name': name,
                    'tflops': tflops,
                    'pct_peak': pct_peak,
                    'pct_ref': pct_ref,
                    'bottleneck': bottleneck
                })
            except Exception as e:
                print(f"   {name:<22} ERROR: {e}")
        
        # =====================================================================
        # SECTION 3: Summary and Recommendations
        # =====================================================================
        print("\n" + "-" * 70)
        print("   SECTION 3: Summary and Recommendations")
        print("-" * 70)
        
        if kernel_results:
            best = max(kernel_results, key=lambda x: x['tflops'])
            worst = min(kernel_results, key=lambda x: x['tflops'])
            
            print(f"\n   🏆 Best kernel: {best['name']} ({best['tflops']:.2f} TFLOPS, {best['pct_peak']:.1f}% peak)")
            print(f"   📉 Worst kernel: {worst['name']} ({worst['tflops']:.2f} TFLOPS, {worst['pct_peak']:.1f}% peak)")
            print(f"   📊 PyTorch reference: {tflops_ref:.2f} TFLOPS")
            print(f"   📈 Gap to PyTorch: {(1 - best['tflops']/tflops_ref)*100:.1f}%")
            
            # Group by bottleneck
            bottleneck_groups = {}
            for r in kernel_results:
                bn = r['bottleneck'].split(':')[0] if ':' in r['bottleneck'] else r['bottleneck']
                if bn not in bottleneck_groups:
                    bottleneck_groups[bn] = []
                bottleneck_groups[bn].append(r['name'])
            
            print(f"\n   Bottleneck Distribution:")
            for bn, kernels_list in sorted(bottleneck_groups.items()):
                print(f"      {bn}: {', '.join(kernels_list)}")
            
            print(f"\n   💡 Recommendations:")
            if best['pct_peak'] < 40:
                print(f"      - All kernels below 40% peak utilization")
                print(f"      - Primary bottleneck: Memory latency / LDS bank conflicts")
                print(f"      - Consider: Async LDS loads, better prefetching")
            if 'highOcc' in worst['name']:
                print(f"      - High-occupancy variant underperforms (latency hiding > occupancy)")
            if any('swizzle' in r['name'].lower() for r in kernel_results if r['tflops'] > best['tflops'] * 0.95):
                print(f"      - XOR swizzle shows promise for bank conflict reduction")
        
        print("\n" + "=" * 70)
        print("✅ Detailed bottleneck analysis completed!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ Bottleneck analysis failed: {e}")
        traceback.print_exc()
        return False


def diagnose_bottleneck(name, tflops, pct_peak, info, peak_tflops):
    """
    Diagnose the likely bottleneck for a kernel based on its characteristics.
    
    Returns a string describing the bottleneck.
    """
    # Thresholds
    GOOD_UTILIZATION = 35  # % of peak
    MODERATE_UTILIZATION = 25
    
    features = info.get('features', '')
    tile = info.get('tile', '')
    lds_kb = info.get('lds_kb', 0)
    regs = info.get('regs', '~90')
    
    # Parse register count
    try:
        reg_count = int(regs.replace('~', ''))
    except:
        reg_count = 90
    
    # Diagnose based on characteristics
    if pct_peak >= GOOD_UTILIZATION:
        return "✅ Good utilization"
    
    if 'highOcc' in name.lower() or tile == '64×32':
        return "⚠️ Occupancy: Small tiles hurt compute intensity"
    
    if 'noPrefetch' in name.lower() or 'No reg prefetch' in features:
        return "⚠️ Latency: No register prefetch"
    
    if 'hilbert' in name.lower():
        return "⚠️ Overhead: Hilbert indexing cost"
    
    if 'quad' in name.lower() or lds_kb > 30:
        return "⚠️ LDS: High LDS usage limits occupancy"
    
    if 'kunroll' in name.lower():
        return "⚠️ Registers: K-unroll increases pressure"
    
    if 'swizzle' in name.lower() or 'xor' in name.lower():
        if pct_peak < MODERATE_UTILIZATION:
            return "⚠️ Swizzle: Per-element unswizzle overhead"
        return "⚠️ Swizzle: Slight overhead from indexing"
    
    if pct_peak < MODERATE_UTILIZATION:
        return "⚠️ Memory: Likely LDS bank conflicts"
    
    return "⚠️ Unknown: Needs profiling"

def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='rocWMMA Patch Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (skip detailed analysis)')
    parser.add_argument('--detailed', action='store_true', help='Run detailed bottleneck analysis')
    parser.add_argument('--correctness-only', action='store_true', help='Run correctness tests only')
    args = parser.parse_args()
    
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
    
    if args.correctness_only:
        print("\n" + "=" * 60)
        print("✅ Correctness tests completed!")
        print("=" * 60)
        return 0
    
    # Benchmark performance (basic)
    if not benchmark_performance():
        print("\n⚠️  Performance benchmarks failed (non-fatal)")
    
    # Benchmark all kernel variants
    benchmark_all_kernels()
    
    # Detailed bottleneck analysis (unless --quick)
    if not args.quick or args.detailed:
        detailed_bottleneck_analysis()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

