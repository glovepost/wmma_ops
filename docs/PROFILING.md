# GPU Profiling Guide for WMMA Kernels

This guide covers profiling and debugging tools for analyzing WMMA kernel performance on AMD gfx1151 (RDNA3.5).

## Quick Start

```bash
# Build the profiling container
docker compose -f docker/docker-compose.profiling.yml build

# Run environment check
docker compose -f docker/docker-compose.profiling.yml run profiling

# Profile WMMA kernels
docker compose -f docker/docker-compose.profiling.yml run profile

# Collect hardware counters
docker compose -f docker/docker-compose.profiling.yml run counters

# Generate HIP API traces
docker compose -f docker/docker-compose.profiling.yml run trace
```

## Available Tools

### 1. rocprofv3 (Recommended for ROCm 7.x)

The new profiling infrastructure for ROCm. Provides:
- Hardware counter collection
- HIP API tracing
- Kernel dispatch tracing
- PC sampling

**Basic Usage:**
```bash
# Inside container
rocprofv3 --hip-trace python your_script.py

# With hardware counters
rocprofv3 --counter SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_LDS python your_script.py
```

### 2. rocprof (Legacy, ROCm 6.x)

Older profiling tool, still useful for some metrics:
```bash
rocprof --hip-trace --hsa-trace -o output.csv python your_script.py
```

### 3. Omnitrace

Comprehensive tracing with timeline visualization:
```bash
omnitrace-python -- python your_script.py
# View output in Perfetto UI (https://ui.perfetto.dev)
```

### 4. Radeon GPU Profiler (RGP)

Best for RDNA3 detailed analysis. Requires:
- Radeon Developer Panel on host
- RGP GUI for analysis

Features:
- Instruction-level timing
- Wavefront occupancy visualization
- Memory counter analysis
- ISA disassembly

## ⚠️ Important: gfx1151 Profiling Limitations

**Hardware profiling counters are NOT available on gfx1151 consumer GPUs.**

The `aqlprofile` API used by rocprof/rocprofv3 for hardware counter collection is not supported on RDNA3 consumer GPUs. You will see this error if you try:
```
aqlprofile API table load failed: HSA_STATUS_ERROR: A generic error has occurred.
```

**Available profiling methods on gfx1151:**
- ✅ Timing-based profiling (wall clock)
- ✅ PyTorch profiler (torch.profiler)
- ✅ Basic HIP API tracing
- ❌ Hardware counters (SQ_WAVES, TCC_HIT, etc.)
- ❌ Instruction-level profiling
- ❌ rocprof-compute/Omniperf roofline analysis

For hardware counter access, you need AMD Instinct (MI-series) GPUs or use Radeon GPU Profiler (RGP) on Windows with special driver support.

## Key Metrics for WMMA Kernels (Reference)

These metrics are useful on supported hardware (MI-series):

| Metric | Description | Target |
|--------|-------------|--------|
| `SQ_WAVES` | Total wavefronts launched | Higher = more parallelism |
| `SQ_INSTS_VALU` | Vector ALU instructions | Should dominate |
| `SQ_INSTS_LDS` | LDS instructions | Monitor for bank conflicts |
| `SQ_WAIT_INST_LDS` | Cycles waiting on LDS | Should be low |
| `TCC_HIT` / `TCC_MISS` | L2 cache hit/miss | High hit rate desired |
| `SQ_VALU_MFMA_MOPS` | Matrix multiply ops | Core WMMA metric |

## Profiling Workflow

### Step 1: Basic Timing
```bash
docker compose -f docker/docker-compose.profiling.yml run profile
```

This runs `profile_wmma.py` which benchmarks all kernel variants.

### Step 2: Identify Bottlenecks
```bash
docker compose -f docker/docker-compose.profiling.yml run counters
```

Collects hardware counters to identify:
- Memory bandwidth limitations
- LDS bank conflicts
- Instruction mix issues

### Step 3: Detailed Trace Analysis
```bash
docker compose -f docker/docker-compose.profiling.yml run trace
```

Generates HIP API traces showing:
- Kernel launch timing
- Memory transfer overhead
- Synchronization points

## Counter Files

Pre-configured counter sets in `/workspace/profiling/`:

### `counters_basic.txt`
Essential counters for quick analysis:
- Wavefront counts
- Instruction mix (VALU, LDS, VMEM)
- Basic cache metrics

### `counters_extended.txt`
Detailed counters for deep analysis:
- LDS bank conflicts
- Full cache hierarchy
- Memory controller metrics

## Interpreting Results

### Good WMMA Kernel Profile
```
SQ_WAVES:        high (good parallelism)
SQ_INSTS_VALU:   dominant (compute-bound)
SQ_INSTS_LDS:    moderate (data staging)
SQ_WAIT_INST_LDS: low (no bank conflicts)
TCC_HIT:         high (good cache reuse)
```

### Problem Indicators
- **High `SQ_WAIT_INST_LDS`**: LDS bank conflicts, consider padding
- **High `TCC_MISS`**: Poor cache locality, check tile sizes
- **Low `SQ_WAVES`**: Occupancy limited, reduce register/LDS usage
- **High `SQ_INSTS_VMEM`**: Memory-bound, improve data reuse

## Output Files

Traces are saved to `/workspace/traces/` (mounted to `./traces/` on host):

```
traces/
├── trace_20251228_120000/      # HIP trace output
│   ├── hip_api_trace.json
│   └── kernel_trace.json
├── counters_20251228_120100.csv  # Hardware counters
└── wmma_profile_20251228_120200.json  # Timing results
```

## Troubleshooting

### "Permission denied" for profiling
Ensure container has `SYS_PTRACE` and `SYS_ADMIN` capabilities (set in docker-compose).

### "Counter not available"
Some counters are architecture-specific. Use `rocprofv3 --list-counters` to see available counters for gfx1151.

### "rocprofv3 not found"
The container uses ROCm 6.3 APT packages for profiling tools. ROCm 7.x pip packages don't include profiling binaries. Use `rocprof` instead.

### Inconsistent timing results
Disable TunableOp during profiling:
```bash
export PYTORCH_TUNABLEOP_ENABLED=0
```

## References

- [ROCm Profiling Tools Blog](https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html)
- [rocprofiler-sdk Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [Radeon GPU Profiler](https://gpuopen.com/rgp/)
- [Omnitrace GitHub](https://github.com/ROCm/omnitrace)
