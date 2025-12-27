#!/usr/bin/env python3
"""
Advanced profiling for WMMA kernel using available ROCm tools.

This script provides multiple profiling methods:
1. PyTorch profiler with CUDA events
2. HIP events for precise kernel timing
3. Memory bandwidth analysis
4. Compute utilization estimation

Since rocprof may not be available in all environments, this script
uses alternative profiling methods that work with PyTorch/ROCm.
"""

import torch
import time
import sys
import os

# Set up environment
tunableop_dir = "/tmp/tunableop"
os.makedirs(tunableop_dir, mode=0o777, exist_ok=True)
if "TUNABLEOP_RESULTS_DIR" not in os.environ:
    os.environ["TUNABLEOP_RESULTS_DIR"] = tunableop_dir

# Set library path
try:
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{current_ld_path}"
except:
    pass

import wmma_ops


def profile_with_events():
    """Profile using CUDA/HIP events for precise timing."""
    print("=" * 70)
    print("WMMA Kernel Profiling with HIP Events")
    print("=" * 70)
    
    device = "cuda"
    
    # Test configurations
    configs = [
        (2048, 2048, 64, "Small K"),
        (4096, 4096, 128, "Medium K"),
        (4096, 4096, 2048, "Large K"),
    ]
    
    print(f"\n{'Configuration':<30} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'Util %':<8} | {'BW (GB/s)':<10}")
    print("-" * 85)
    
    results = []
    
    for M, N, K, label in configs:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = wmma_ops.matmul(A, B)
        torch.cuda.synchronize()
        
        # Create events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        iterations = 100
        
        # Profile
        start_event.record()
        for _ in range(iterations):
            C = wmma_ops.matmul(A, B)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # Calculate metrics
        elapsed_ms = start_event.elapsed_time(end_event) / iterations
        ops = 2 * M * N * K
        tflops = (ops / (elapsed_ms * 1e-3)) / 1e12
        utilization = (tflops / 59.4) * 100  # Peak gfx1151: 59.4 TFLOPS
        
        # Memory bandwidth: A (M×K×2 bytes) + B (K×N×2 bytes) + C (M×N×4 bytes)
        bytes_transferred = (M * K * 2) + (K * N * 2) + (M * N * 4)
        bandwidth_gbs = (bytes_transferred / (elapsed_ms * 1e-3)) / 1e9
        
        results.append({
            'config': label,
            'M': M, 'N': N, 'K': K,
            'time_ms': elapsed_ms,
            'tflops': tflops,
            'utilization': utilization,
            'bandwidth_gbs': bandwidth_gbs
        })
        
        print(f"{label} ({M}×{N}×{K})".ljust(30) + 
              f" | {elapsed_ms:>10.3f} | {tflops:>8.2f} | {utilization:>6.1f}% | {bandwidth_gbs:>8.1f}")
    
    return results


def profile_with_pytorch_profiler():
    """Profile using PyTorch's built-in profiler for detailed breakdown."""
    print("\n" + "=" * 70)
    print("PyTorch Profiler Analysis")
    print("=" * 70)
    
    device = "cuda"
    M, N, K = 4096, 4096, 2048
    
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(5):
        _ = wmma_ops.matmul(A, B)
    torch.cuda.synchronize()
    
    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True
    ) as prof:
        for _ in range(10):
            C = wmma_ops.matmul(A, B)
        torch.cuda.synchronize()
    
    print("\n📊 Kernel Breakdown:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # Export trace
    trace_file = '/tmp/wmma_trace.json'
    try:
        prof.export_chrome_trace(trace_file)
        print(f"\n📄 Chrome trace exported to: {trace_file}")
        print("   View in Chrome: chrome://tracing")
    except Exception as e:
        print(f"   ⚠️ Could not export trace: {e}")
    
    return prof


def analyze_memory_patterns():
    """Analyze memory access patterns and bandwidth utilization."""
    print("\n" + "=" * 70)
    print("Memory Access Pattern Analysis")
    print("=" * 70)
    
    device = "cuda"
    
    # Test with different matrix sizes to understand memory behavior
    configs = [
        (1024, 1024, 1024, "1K×1K×1K (Compute-bound)"),
        (4096, 4096, 64, "4K×4K×64 (Memory-bound)"),
        (4096, 4096, 2048, "4K×4K×2K (Balanced)"),
        (8192, 8192, 4096, "8K×8K×4K (Large)"),
    ]
    
    print(f"\n{'Configuration':<35} | {'Arithmetic':<15} | {'Memory':<15} | {'Intensity':<10}")
    print(f"{'':35} | {'Intensity':15} | {'(MB)':15} | {'(FLOPS/B)':<10}")
    print("-" * 85)
    
    for M, N, K, label in configs:
        # Calculate arithmetic intensity
        flops = 2 * M * N * K
        # Memory: A (M×K×2) + B (K×N×2) + C (M×N×4)
        memory_bytes = (M * K * 2) + (K * N * 2) + (M * N * 4)
        memory_mb = memory_bytes / (1024 * 1024)
        intensity = flops / memory_bytes
        
        print(f"{label:<35} | {flops/1e9:>13.2f}G | {memory_mb:>13.1f} | {intensity:>8.1f}")
    
    print("\n📊 Analysis:")
    print("- Arithmetic Intensity < 10: Memory-bound")
    print("- Arithmetic Intensity 10-50: Balanced")
    print("- Arithmetic Intensity > 50: Compute-bound")
    print("- gfx1151 optimal: ~59.4 TFLOPS / 256 GB/s = ~232 FLOPS/byte")


def compare_with_pytorch():
    """Compare WMMA kernel with PyTorch's default implementation."""
    print("\n" + "=" * 70)
    print("WMMA vs PyTorch Comparison")
    print("=" * 70)
    
    device = "cuda"
    M, N, K = 4096, 4096, 2048
    
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    
    iterations = 50
    
    # Warmup
    for _ in range(10):
        _ = wmma_ops.matmul(A, B)
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Profile WMMA
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        C_wmma = wmma_ops.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    wmma_time = start_event.elapsed_time(end_event) / iterations
    
    # Profile PyTorch
    start_event.record()
    for _ in range(iterations):
        C_ref = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / iterations
    
    # Calculate TFLOPS
    ops = 2 * M * N * K
    wmma_tflops = (ops / (wmma_time * 1e-3)) / 1e12
    pytorch_tflops = (ops / (pytorch_time * 1e-3)) / 1e12
    
    print(f"\n📊 Results for {M}×{N}×{K} matrix multiplication:")
    print(f"\n{'Kernel':<15} | {'Time (ms)':<12} | {'TFLOPS':<10} | {'Utilization':<12}")
    print("-" * 55)
    print(f"{'WMMA':<15} | {wmma_time:>10.3f} | {wmma_tflops:>8.2f} | {(wmma_tflops/59.4)*100:>10.1f}%")
    print(f"{'PyTorch':<15} | {pytorch_time:>10.3f} | {pytorch_tflops:>8.2f} | {(pytorch_tflops/59.4)*100:>10.1f}%")
    print(f"\n📈 Speedup: {pytorch_time/wmma_time:.2f}x {'faster' if wmma_time < pytorch_time else 'slower'}")
    
    # Correctness check
    max_err = (C_wmma - C_ref).abs().max().item()
    print(f"📋 Max error: {max_err:.6f}")


def check_rocprof_availability():
    """Check if rocprof is available and provide installation instructions."""
    print("\n" + "=" * 70)
    print("rocprof Availability Check")
    print("=" * 70)
    
    import subprocess
    
    # Check for rocprofv2 (required for gfx1151/RDNA3.5)
    paths_to_check = [
        ('/opt/rocm/bin/rocprofv2', 'rocprofv2'),
        ('/opt/rocm/bin/rocprof', 'rocprof v1'),
    ]
    
    rocprof_found = False
    for path, name in paths_to_check:
        try:
            result = subprocess.run([path, '--version'] if 'v2' in name else [path, '--help'], 
                                  capture_output=True, text=True, timeout=5)
            print(f"✅ {name} found at: {path}")
            rocprof_found = True
        except:
            print(f"❌ {name} not found at: {path}")
            continue
    
    print("\n⚠️  Note: rocprof v1 is NOT supported on gfx1151 (RDNA3.5)")
    print("   rocprofv2 is required, but may conflict with PyTorch ROCm initialization")
    print("   Using PyTorch profiler as the primary profiling method")
    
    print("\n📦 Alternative profiling options:")
    print("1. PyTorch profiler (used in this script) - Works with WMMA kernel")
    print("2. AMD Radeon GPU Profiler (RGP) GUI:")
    print("   https://github.com/GPUOpen-Tools/radeon_gpu_profiler/releases")
    print("3. rocprof-compute (requires additional dependencies):")
    print("   pip install astunparse colorlover dash matplotlib pymongo tabulate")
    
    return rocprof_found


def run_rocprof_if_available():
    """Note about rocprof limitations on gfx1151."""
    print("\n" + "=" * 70)
    print("rocprof Status for gfx1151")
    print("=" * 70)
    
    print("""
⚠️  rocprof v1 is NOT supported on gfx1151 (RDNA3.5)

The error "rocprof(v1) is not supported on this device" occurs because
RDNA3.5 requires rocprofv2, which has library conflicts with PyTorch's
ROCm initialization.

📊 Current profiling uses PyTorch's built-in profiler, which provides:
   - Kernel execution times
   - Memory bandwidth analysis
   - Compute utilization estimates
   - Bottleneck identification

🔧 For more detailed profiling, consider:
   1. AMD Radeon GPU Profiler (RGP) - GUI-based profiler
      https://github.com/GPUOpen-Tools/radeon_gpu_profiler/releases
   
   2. rocprof-compute with dependencies:
      pip install astunparse colorlover dash matplotlib pymongo tabulate
      rocprof-compute profile -- python your_script.py
   
   3. Direct rocprofv2 in a separate process (without PyTorch)
      /opt/rocm/bin/rocprofv2 --kernel-trace -- <binary>
""")
    return False


def bottleneck_analysis(results):
    """Analyze profiling results and identify bottlenecks."""
    print("\n" + "=" * 70)
    print("🔍 BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    if not results:
        print("No profiling results to analyze")
        return
    
    avg_tflops = sum(r['tflops'] for r in results) / len(results)
    avg_util = sum(r['utilization'] for r in results) / len(results)
    avg_bw = sum(r['bandwidth_gbs'] for r in results) / len(results)
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Average TFLOPS: {avg_tflops:.2f}")
    print(f"   Average Utilization: {avg_util:.1f}%")
    print(f"   Average Bandwidth: {avg_bw:.1f} GB/s")
    
    print("\n🎯 Bottleneck Identification:")
    
    # Determine primary bottleneck
    peak_tflops = 59.4  # gfx1151 peak
    peak_bw = 256  # LPDDR5X peak GB/s
    
    compute_efficiency = avg_tflops / peak_tflops
    memory_efficiency = avg_bw / peak_bw
    
    print(f"   Compute Efficiency: {compute_efficiency*100:.1f}%")
    print(f"   Memory Efficiency: {memory_efficiency*100:.1f}%")
    
    if compute_efficiency < 0.15:
        print("\n🔴 PRIMARY BOTTLENECK: LOW COMPUTE UTILIZATION")
        print("   - Kernel is memory-bound (waiting for data)")
        print("   - Recommendations:")
        print("     1. Increase arithmetic intensity (larger K)")
        print("     2. Implement register blocking (2×2 tiles)")
        print("     3. Optimize LDS banking (A_STRIDE = 64)")
        print("     4. Use GMEM load spreading")
    
    if memory_efficiency < 0.5:
        print("\n🟡 SECONDARY BOTTLENECK: LOW MEMORY BANDWIDTH")
        print("   - Memory access patterns are inefficient")
        print("   - Recommendations:")
        print("     1. Use vectorized loads (half8)")
        print("     2. Improve data reuse in LDS")
        print("     3. Optimize global memory access coalescing")
    
    print("\n📈 Expected Improvements from Optimizations:")
    print(f"   Current: {avg_tflops:.2f} TFLOPS ({avg_util:.1f}%)")
    print(f"   + Register Blocking: {avg_tflops * 1.15:.2f} TFLOPS (+15%)")
    print(f"   + LDS Banking: {avg_tflops * 1.15 * 1.10:.2f} TFLOPS (+10%)")
    print(f"   + GMEM Spreading: {avg_tflops * 1.15 * 1.10 * 1.10:.2f} TFLOPS (+10%)")
    print(f"   Target: {avg_tflops * 1.15 * 1.10 * 1.10:.2f} TFLOPS ({avg_util * 1.39:.1f}%)")


def main():
    """Main profiling entry point."""
    print("=" * 70)
    print("WMMA Kernel Profiling Suite")
    print("Target: gfx1151 (RDNA3.5)")
    print("=" * 70)
    
    # Check PyTorch/CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available")
        return
    
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Device: {torch.cuda.get_device_name(0)}")
    
    # Run profiling methods
    results = profile_with_events()
    profile_with_pytorch_profiler()
    analyze_memory_patterns()
    compare_with_pytorch()
    
    # Check and run rocprof if available
    check_rocprof_availability()
    run_rocprof_if_available()
    
    # Analyze bottlenecks
    bottleneck_analysis(results)
    
    print("\n" + "=" * 70)
    print("✅ Profiling Complete")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Profiling interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

