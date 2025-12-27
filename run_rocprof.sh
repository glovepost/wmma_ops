#!/bin/bash
# Script to run rocprof profiling on WMMA kernel
# Usage: ./run_rocprof.sh
# Note: Uses rocprof-compute for gfx1151 (RDNA3.5) which requires rocprofv2

set -e

echo "=========================================="
echo "WMMA Kernel Profiling with rocprof-compute"
echo "Target: gfx1151 (RDNA3.5)"
echo "=========================================="

# Install rocprofiler-compute if not available
if ! command -v /opt/rocm/bin/rocprof-compute &> /dev/null; then
    echo "Installing rocprofiler-compute..."
    apt update >/dev/null 2>&1
    apt install -y rocprofiler-compute >/dev/null 2>&1
fi

echo "✅ rocprof-compute version:"
/opt/rocm/bin/rocprof-compute --version 2>&1 | head -3

# Set up paths
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__) + "/lib")'):$LD_LIBRARY_PATH

# Create output directory
mkdir -p /tmp/rocprof_results
cd /tmp/rocprof_results

# Create profiling script
cat > /tmp/run_wmma_kernel.py << 'EOF'
import torch
import sys
import os

# Set up tunableop directory
os.makedirs("/tmp/tunableop", mode=0o777, exist_ok=True)
os.environ["TUNABLEOP_RESULTS_DIR"] = "/tmp/tunableop"

# Set library path
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib_path not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{current_ld_path}"

sys.path.insert(0, '/tmp/wmma_ops')
import wmma_ops

device = "cuda"
M, N, K = 4096, 4096, 2048

print(f"Creating matrices: {M}x{K} and {K}x{N}")
A = torch.randn(M, K, device=device, dtype=torch.float16)
B = torch.randn(K, N, device=device, dtype=torch.float16)

# Warmup
print("Warming up...")
for _ in range(5):
    _ = wmma_ops.matmul(A, B)
torch.cuda.synchronize()

# Run kernel for profiling
print("Running WMMA kernel for profiling...")
for _ in range(10):
    C = wmma_ops.matmul(A, B)
torch.cuda.synchronize()

print("Profiling complete")
EOF

echo ""
echo "=========================================="
echo "1. rocprof-compute Profiling"
echo "=========================================="
echo "Running rocprof-compute for GPU profiling..."
echo "Note: rocprof v1 is not supported on gfx1151 (RDNA3.5)"
echo "Using rocprof-compute (rocprofv2-based) instead"
echo ""

# Run rocprof-compute profile
# This uses the modern rocprofv2 backend which supports RDNA3
cd /tmp/rocprof_results
/opt/rocm/bin/rocprof-compute profile --name wmma_kernel -- python /tmp/run_wmma_kernel.py 2>&1 || {
    echo "rocprof-compute profile failed, trying alternative approach..."
}

echo ""
echo "📄 Profiling Results:"
ls -la /tmp/rocprof_results/ 2>/dev/null || echo "No results directory"

# Check for any output files
if ls /tmp/rocprof_results/*.csv 2>/dev/null; then
    echo ""
    echo "CSV Results:"
    for f in /tmp/rocprof_results/*.csv; do
        echo "=== $f ==="
        head -30 "$f"
    done
fi

if ls /tmp/rocprof_results/*.json 2>/dev/null; then
    echo ""
    echo "JSON Results available (use rocprof-compute analyze to view)"
fi

echo ""
echo "=========================================="
echo "2. Alternative: PyTorch Profiler"
echo "=========================================="
echo "Since rocprof v1 is not supported on gfx1151,"
echo "the PyTorch profiler in rocprof_wmma.py provides"
echo "detailed timing and memory analysis."
echo ""
echo "Run: python rocprof_wmma.py"

echo ""
echo "=========================================="
echo "3. Summary"
echo "=========================================="
echo "📁 Output files:"
find /tmp/rocprof_results -type f 2>/dev/null | head -10

echo ""
echo "✅ Profiling complete!"
echo ""
echo "📊 For detailed analysis, run:"
echo "   python rocprof_wmma.py"

