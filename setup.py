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

setup(
    name='wmma_ops',
    ext_modules=[
        CppExtension(
            name='wmma_ops',
            sources=['wmma_gemm.hip'],
            extra_compile_args=[
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
            ],
            include_dirs=[patch_dir],  # Also specify in include_dirs for PyTorch
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
