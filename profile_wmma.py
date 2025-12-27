#!/usr/bin/env python3
"""
Profile WMMA kernel using rocprof to identify performance bottlenecks.

This script runs rocprof to collect performance metrics including:
- Kernel execution time
- Memory bandwidth utilization
- Instruction counts
- VALU utilization
- LDS usage
- Register usage
"""

import torch
import wmma_ops
import subprocess
import os
import json
import sys

def create_profiling_config():
    """Create rocprof configuration file for detailed metrics."""
    config = """
# rocprof configuration for WMMA kernel profiling
pmc: SQ_WAVES, SQ_INSTS_VALU, SQ_INSTS_SALU, SQ_INSTS_SMEM, SQ_INSTS_VMEM, SQ_INSTS_FLAT
# Memory metrics
pmc: SQ_INSTS_VMEM_LDS, SQ_INSTS_VMEM_GDS, SQ_INSTS_VMEM_GLOBAL
# Compute metrics
pmc: SQ_INSTS_VALU_FP16, SQ_INSTS_VALU_FP32, SQ_INSTS_VALU_FP64
# Wavefront metrics
pmc: SQ_WAVES_ACTIVE, SQ_WAVES_SLEEPING
# LDS metrics
pmc: SQ_LDS_BANK_CONFLICT
"""
    with open('/tmp/rocprof_config.txt', 'w') as f:
        f.write(config)
    return '/tmp/rocprof_config.txt'

def run_profiling():
    """Run WMMA kernel with rocprof profiling."""
    print("=" * 60)
    print("WMMA Kernel Profiling with rocprof")
    print("=" * 60)
    
    # Check if rocprof is available
    try:
        result = subprocess.run(['rocprof', '--version'], 
                              capture_output=True, text=True, timeout=5)
        print(f"✅ rocprof found: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ rocprof not found. Trying alternative profiling methods...")
        return run_basic_profiling()
    
    # Create test matrices
    device = "cuda"
    M, N, K = 4096, 4096, 2048
    
    print(f"\n📊 Profiling matrix multiplication: {M}×{K} × {K}×{N}")
    
    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(K, N, device=device, dtype=torch.float16)
    
    # Warmup
    print("🔥 Warming up...")
    for _ in range(5):
        _ = wmma_ops.matmul(A, B)
    torch.cuda.synchronize()
    
    # Create rocprof command
    config_file = create_profiling_config()
    
    # Create a simple Python script to run the kernel
    script_content = f"""
import torch
import wmma_ops
import sys

device = "cuda"
M, N, K = {M}, {N}, {K}
A = torch.randn(M, K, device=device, dtype=torch.float16)
B = torch.randn(K, N, device=device, dtype=torch.float16)

# Run kernel
C = wmma_ops.matmul(A, B)
torch.cuda.synchronize()
print("Kernel execution completed")
"""
    
    script_path = '/tmp/run_wmma_kernel.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Run with rocprof
    print("\n🔍 Running rocprof...")
    try:
        cmd = [
            'rocprof',
            '--stats',  # Generate statistics
            '--timestamp',  # Include timestamps
            f'--config={config_file}',
            'python', script_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Profiling completed successfully")
            print("\n" + "=" * 60)
            print("PROFILING RESULTS")
            print("=" * 60)
            print(result.stdout)
            if result.stderr:
                print("\nWarnings/Errors:")
                print(result.stderr)
            
            # Try to find and display the results file
            import glob
            csv_files = glob.glob('results_*.csv')
            if csv_files:
                print(f"\n📄 Results saved to: {csv_files[0]}")
                with open(csv_files[0], 'r') as f:
                    print("\nFirst 50 lines of results:")
                    for i, line in enumerate(f):
                        if i < 50:
                            print(line.rstrip())
                        else:
                            break
        else:
            print("❌ rocprof failed:")
            print(result.stderr)
            return run_basic_profiling()
            
    except subprocess.TimeoutExpired:
        print("⏱️  rocprof timed out, falling back to basic profiling")
        return run_basic_profiling()
    except Exception as e:
        print(f"❌ Error running rocprof: {e}")
        return run_basic_profiling()

def run_basic_profiling():
    """Fallback: Basic profiling using PyTorch's built-in profiler."""
    print("\n" + "=" * 60)
    print("Basic Profiling (PyTorch Profiler)")
    print("=" * 60)
    
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
        with_stack=True
    ) as prof:
        for _ in range(10):
            C = wmma_ops.matmul(A, B)
        torch.cuda.synchronize()
    
    print("\n📊 Profiling Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Export to Chrome trace format
    trace_file = '/tmp/wmma_profiler_trace.json'
    prof.export_chrome_trace(trace_file)
    print(f"\n📄 Chrome trace saved to: {trace_file}")
    print("   Open in Chrome: chrome://tracing")

def analyze_bottlenecks():
    """Analyze profiling results and provide recommendations."""
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    print("""
Based on the Deep Dive guide and expected bottlenecks:

1. **Memory Bandwidth** (Most Likely)
   - Check: SQ_INSTS_VMEM_GLOBAL vs SQ_INSTS_VALU
   - If VMEM >> VALU: Memory-bound
   - Solution: GMEM load spreading (Kernel 8 optimization)

2. **LDS Bank Conflicts**
   - Check: SQ_LDS_BANK_CONFLICT
   - If conflicts > 0: Need better LDS padding
   - Solution: A_STRIDE = 64 (128-byte alignment)

3. **VALU Utilization**
   - Check: SQ_INSTS_VALU / SQ_WAVES
   - If low: Not using dual-issue effectively
   - Solution: Instruction scheduling optimization

4. **Register Pressure**
   - Check: Register usage per thread
   - If > 128 VGPRs: May limit occupancy
   - Solution: Optimize register usage

5. **Wavefront Occupancy**
   - Check: SQ_WAVES_ACTIVE vs SQ_WAVES_SLEEPING
   - If sleeping > active: Low occupancy
   - Solution: Adjust block size and launch bounds
""")

if __name__ == "__main__":
    try:
        run_profiling()
        analyze_bottlenecks()
    except KeyboardInterrupt:
        print("\n\n⚠️  Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)






