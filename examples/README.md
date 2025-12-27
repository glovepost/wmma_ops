# WMMA Reference Materials

Reference implementations and examples for optimizing WMMA GEMM kernels on AMD GPUs.

## Contents

### 1. adelj_wmma_samples/
Full clone of [adelj88/rocm_wmma_samples](https://github.com/adelj88/rocm_wmma_samples)
- `hgemm/` - Half-precision GEMM implementations with XOR swizzle, multi-level tiling
- Includes examples of LDS bank conflict avoidance techniques

### 2. awesome-gemm/
Full clone of [yuninxia/awesome-gemm](https://github.com/yuninxia/awesome-gemm)
- Curated list of GEMM optimization resources
- Links to papers, implementations, and optimization techniques

### 3. rocwmma_samples/
Selected files from [ROCm/rocWMMA](https://github.com/ROCm/rocWMMA)
- `perf_hgemm.cpp` - Official rocWMMA performance sample

### 4. rocm_examples_sparse/
Sparse checkout of [ROCm/rocm-examples](https://github.com/ROCm/rocm-examples)
- `HIP-Basic/` - Basic HIP programming examples
- `LLVM-AMDGPU/` - AMDGPU-specific LLVM examples

### 5. rocmlir_docs/
Selected documentation from [ROCm/rocMLIR](https://github.com/ROCm/rocMLIR)
- MLIR-based GPU dialect for AMD GPUs

### 6. composable_kernel_issue/
- `issue_1434.md` - CK issue discussing WMMA on RDNA3

## Key References Online

- [Composable Kernel WMMA Issue #1434](https://github.com/ROCm/composable_kernel/issues/1434)
- [rocWMMA Library](https://github.com/ROCm/rocWMMA)
- [Deep Dive into Matrix Optimization on AMD GPUs](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html)
