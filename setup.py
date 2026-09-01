from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Ensure we are compiling for the correct architecture
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1151"

# Use hipcc as the compiler (matching wmma_direct pattern)
rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
os.environ['CXX'] = f'{rocm_path}/bin/hipcc'
os.environ['CC'] = f'{rocm_path}/bin/hipcc'

# Get the directory containing this setup.py
setup_dir = os.path.dirname(os.path.abspath(__file__))
patch_dir = os.path.join(setup_dir, 'rocwmma_patch')

extra_compile_args = [
    '-DAMDGPU_TARGETS=gfx1151',
    '-D__gfx1151__',  # Enable gfx1151-specific optimizations
    f'-I{rocm_path}/include',
    f'-I{rocm_path}/include/rocwmma',
    f'-I{patch_dir}',  # Include rocwmma_patch directory
    '-std=c++17',
    '-O3',  # Enable optimizations
    '--offload-arch=gfx1151',  # CRITICAL: Enables Feature1_5xVGPRs and SALU FP
    '-DCUDA_HAS_FP16=1',
    '-D__HIP_PLATFORM_AMD__=1',
    '-fPIC',
    '-save-temps',  # Save intermediate files for ISA inspection
]

opt_min_blocks = os.environ.get('WMMA_OPT_MIN_BLOCKS_PER_CU')
if opt_min_blocks is not None:
    try:
        opt_min_blocks_value = int(opt_min_blocks)
    except ValueError as error:
        raise ValueError('WMMA_OPT_MIN_BLOCKS_PER_CU must be an integer') from error
    if not 0 <= opt_min_blocks_value <= 4:
        raise ValueError('WMMA_OPT_MIN_BLOCKS_PER_CU must be between 0 and 4')
    extra_compile_args.append(
        f'-DWMMA_OPT_MIN_BLOCKS_PER_CU={opt_min_blocks_value}'
    )

cu_mode = os.environ.get('WMMA_CU_MODE', '0')
if cu_mode not in ('0', '1'):
    raise ValueError('WMMA_CU_MODE must be 0 or 1')
if cu_mode == '1':
    extra_compile_args.append('-mcumode')

unroll_threshold = os.environ.get('WMMA_UNROLL_THRESHOLD')
if unroll_threshold is not None:
    try:
        unroll_threshold_value = int(unroll_threshold)
    except ValueError as error:
        raise ValueError('WMMA_UNROLL_THRESHOLD must be an integer') from error
    if unroll_threshold_value <= 0:
        raise ValueError('WMMA_UNROLL_THRESHOLD must be positive')
    extra_compile_args.extend([
        '-mllvm',
        f'-amdgpu-unroll-threshold-local={unroll_threshold_value}',
    ])

setup(
    name='wmma_ops',
    ext_modules=[
        CppExtension(
            name='wmma_ops',
            sources=['wmma_gemm.hip'],
            extra_compile_args=extra_compile_args,
            include_dirs=[patch_dir],  # Also specify in include_dirs for PyTorch
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
