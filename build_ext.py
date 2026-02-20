import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Only build if CUDA is available in the environment
if os.system("nvcc --version > /dev/null 2>&1") == 0:
    setup(
        name='vramancer_vtp_c',
        ext_modules=[
            CUDAExtension(
                name='vramancer_vtp_c',
                sources=[
                    'csrc/vtp_core.cpp',
                    'csrc/vtp_cuda.cu',
                ],
                extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']}
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
else:
    print("NVCC not found. Skipping C++/CUDA extension build.")
