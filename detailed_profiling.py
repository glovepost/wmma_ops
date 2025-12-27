#!/usr/bin/env python3
"""
Comprehensive profiling and bottleneck analysis for WMMA kernel.

Uses multiple profiling methods:
1. PyTorch Profiler (available)
2. Manual timing analysis
3. Memory bandwidth estimation
4. Compute utilization analysis
"""

import torch
import wmma_ops
import time
import sys
import os

def estimate_memory_bandwidth(M, N, K, duration_ms):
    """Estimate memory bandwidth usage."""
    # For GEMM: Read A (M×K), Read B (K×N), Write C (M×N)
    # All in FP16 (2 bytes)
    bytes_read = (M * K + K * N) * 2  # A and B
    bytes_written = M * N * 4  # C in FP32 (4 bytes)
    total_bytes = bytes_read + bytes_written
    
    bandwidth_gb_s = (total_bytes / (duration_ms / 1000.0)) / (1024**3)
    return bandwidth_gb_s, total_bytes

def analyze_compute_utilization(tflops, peak_tflops=59.4):
    """Analyze compute utilization."""
    utilization = (tflops / peak_tflops) * 100
    return utilization

def profile_kernel_detailed(M, N, K, iterations=100):
    """Detailed profiling of WMMA kernel."""
    print("=" * 70)
    print(f"Detailed Profiling: {M}×{K} × {K}×{N}")
    print("=" * 70)
    
    device = "cuda"
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(10):
        _ = wmma_ops.matmul(A, B)
    torch.cuda.synchronize()
    
    # Detailed timing
    print("\n📊 Timing Analysis:")
    times = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = wmma_ops.matmul(A, B)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
    
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Min:     {min_time:.3f} ms")
    print(f"  Max:     {max_time:.3f} ms")
    print(f"  Std Dev: {std_time:.3f} ms")
    
    # Compute metrics
    ops = 2 * M * N * K
    tflops = (ops / (avg_time / 1000.0)) / 1e12
    peak_tflops = 59.4  # gfx1151 WMMA peak
    utilization = analyze_compute_utilization(tflops, peak_tflops)
    
    print(f"\n⚡ Performance Metrics:")
    print(f"  TFLOPS:        {tflops:.2f}")
    print(f"  Peak TFLOPS:   {peak_tflops:.2f}")
    print(f"  Utilization:  {utilization:.1f}%")
    
    # Memory bandwidth
    bandwidth, total_bytes = estimate_memory_bandwidth(M, N, K, avg_time)
    print(f"\n💾 Memory Analysis:")
    print(f"  Estimated Bandwidth: {bandwidth:.2f} GB/s")
    print(f"  Total Data:          {total_bytes / (1024**2):.2f} MB")
    print(f"  Data per iteration:  {total_bytes / (1024**2):.2f} MB")
    
    # Compare with PyTorch
    print(f"\n🔍 Comparison with PyTorch:")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        C_ref = torch.matmul(A, B)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000 / iterations
    
    pytorch_tflops = (ops / (pytorch_time / 1000.0)) / 1e12
    speedup = pytorch_time / avg_time
    
    print(f"  PyTorch Time:  {pytorch_time:.3f} ms")
    print(f"  PyTorch TFLOPS: {pytorch_tflops:.2f}")
    print(f"  Speedup:        {speedup:.2f}x")
    
    return {
        'time_ms': avg_time,
        'tflops': tflops,
        'utilization': utilization,
        'bandwidth_gb_s': bandwidth,
        'pytorch_tflops': pytorch_tflops,
        'speedup': speedup
    }

def analyze_bottlenecks(results):
    """Analyze profiling results to identify bottlenecks."""
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    tflops = results['tflops']
    utilization = results['utilization']
    bandwidth = results['bandwidth_gb_s']
    
    print(f"\n📈 Current Performance:")
    print(f"  TFLOPS:       {tflops:.2f}")
    print(f"  Utilization:  {utilization:.1f}%")
    print(f"  Bandwidth:    {bandwidth:.2f} GB/s")
    
    # Identify bottlenecks
    bottlenecks = []
    recommendations = []
    
    # 1. Compute utilization
    if utilization < 20:
        bottlenecks.append("❌ CRITICAL: Very low compute utilization (<20%)")
        recommendations.append("""
        🔧 Recommendations:
           - Verify WMMA intrinsics are being used (check ISA)
           - Enable register blocking (2×2 tiles per warp)
           - Check for instruction dual-issue opportunities
           - Profile with rocprof to verify VALU utilization
        """)
    elif utilization < 40:
        bottlenecks.append("⚠️  Low compute utilization (<40%)")
        recommendations.append("""
        🔧 Recommendations:
           - Implement register blocking (expected +10-15% improvement)
           - Optimize instruction scheduling for dual-issue
           - Check LDS bank conflicts
        """)
    elif utilization < 60:
        bottlenecks.append("⚠️  Moderate compute utilization (<60%)")
        recommendations.append("""
        🔧 Recommendations:
           - Implement GMEM load spreading (Kernel 8 optimization)
           - Optimize LDS banking (A_STRIDE = 64)
           - Enable register blocking for larger blocks
        """)
    else:
        bottlenecks.append("✅ Good compute utilization (≥60%)")
    
    # 2. Memory bandwidth
    # Theoretical peak for gfx1151: ~256 GB/s (LPDDR5X)
    # Measured: ~212 GB/s
    if bandwidth > 180:
        bottlenecks.append("⚠️  High memory bandwidth usage (may be memory-bound)")
        recommendations.append("""
        🔧 Recommendations:
           - Implement GMEM load spreading to reduce wave contention
           - Increase block sizes to improve data reuse
           - Consider prefetching to registers
        """)
    elif bandwidth < 50:
        bottlenecks.append("⚠️  Low memory bandwidth usage (compute-bound)")
        recommendations.append("""
        🔧 Recommendations:
           - This is good! Focus on compute optimizations
           - Enable register blocking
           - Optimize instruction scheduling
        """)
    
    # 3. Performance vs PyTorch
    speedup = results['speedup']
    if speedup < 0.5:
        bottlenecks.append("❌ CRITICAL: Much slower than PyTorch (<0.5x)")
        recommendations.append("""
        🔧 Recommendations:
           - Verify kernel is actually being called
           - Check for correctness issues
           - Profile with rocprof to identify major bottlenecks
           - Compare with wmma_direct implementation
        """)
    elif speedup < 1.0:
        bottlenecks.append("⚠️  Slower than PyTorch (<1.0x)")
        recommendations.append("""
        🔧 Recommendations:
           - Implement all optimizations from Deep Dive guide
           - Focus on register blocking and GMEM load spreading
           - Profile with rocprof for detailed metrics
        """)
    
    print("\n🔍 Identified Bottlenecks:")
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"  {i}. {bottleneck}")
    
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(rec)
    
    # Expected improvements
    print("\n📊 Expected Improvements (from Deep Dive guide):")
    print("  Current:        {:.2f} TFLOPS ({:.1f}% utilization)".format(tflops, utilization))
    
    if utilization < 40:
        expected_reg_blocking = tflops * 1.15  # +15% from register blocking
        expected_gmem_spread = expected_reg_blocking * 1.19  # +19% from GMEM spreading
        expected_total = expected_gmem_spread * 1.10  # +10% from other optimizations
        print("  + Reg Blocking: {:.2f} TFLOPS (+15%)".format(expected_reg_blocking))
        print("  + GMEM Spread:  {:.2f} TFLOPS (+19%)".format(expected_gmem_spread))
        print("  + Other Opts:   {:.2f} TFLOPS (+10%)".format(expected_total))
        print("  Target:        {:.2f} TFLOPS ({:.1f}% utilization)".format(
            expected_total, (expected_total / 59.4) * 100))

def run_pytorch_profiler(M, N, K):
    """Run PyTorch profiler for detailed kernel analysis."""
    print("\n" + "=" * 70)
    print("PyTorch Profiler Analysis")
    print("=" * 70)
    
    device = "cuda"
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        _ = wmma_ops.matmul(A, B)
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=torch.profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        for _ in range(10):
            C = wmma_ops.matmul(A, B)
        torch.cuda.synchronize()
    
    print("\n📊 CUDA Kernel Statistics:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
        max_name_column_width=50
    ))
    
    # Export trace
    trace_file = '/tmp/wmma_detailed_trace.json'
    prof.export_chrome_trace(trace_file)
    print(f"\n📄 Chrome trace saved to: {trace_file}")
    print("   Open in Chrome: chrome://tracing")

def main():
    """Main profiling function."""
    print("=" * 70)
    print("WMMA Kernel Comprehensive Profiling")
    print("=" * 70)
    
    # Test different matrix sizes
    test_configs = [
        (2048, 2048, 64, "Small K"),
        (4096, 4096, 128, "Medium K"),
        (4096, 4096, 2048, "Large K"),
    ]
    
    all_results = []
    
    for M, N, K, label in test_configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {label} ({M}×{N}×{K})")
        print(f"{'='*70}")
        
        results = profile_kernel_detailed(M, N, K, iterations=50)
        results['label'] = label
        results['M'] = M
        results['N'] = N
        results['K'] = K
        all_results.append(results)
        
        analyze_bottlenecks(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'TFLOPS':<10} {'Util %':<10} {'BW (GB/s)':<12} {'vs PyTorch':<12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<15} {r['tflops']:<10.2f} {r['utilization']:<10.1f} "
              f"{r['bandwidth_gb_s']:<12.2f} {r['speedup']:<12.2f}x")
    
    # Run PyTorch profiler on largest config
    print("\n" + "=" * 70)
    M, N, K = 4096, 4096, 2048
    run_pytorch_profiler(M, N, K)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)






