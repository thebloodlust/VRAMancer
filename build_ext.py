import os
from setuptools import setup, Extension
import pybind11

# Always build the pure C++ CPU Swarm extensions (Cross-Platform using GCC/Clang/MSVC)
swarm_extension = Extension(
    'swarm_core',
    sources=['csrc/swarm_core.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17'] if os.name != 'nt' else ['/O2', '/std:c++17']
)

# ⚡ Software CXL Extension for native OS RAM->NVMe without Python serialization bottleneck
software_cxl_extension = Extension(
    'software_cxl',
    sources=['csrc/software_cxl.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17'] if os.name != 'nt' else ['/O2', '/std:c++17']
)

ext_modules = [swarm_extension, software_cxl_extension]

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    # Only build VTP if CUDA is available in the environment
    if os.system("nvcc --version > /dev/null 2>&1") == 0 or os.name == 'nt':
        ext_modules.append(
            CUDAExtension(
                name='vramancer_vtp_c',
                sources=[
                    'csrc/vtp_core.cpp',
                    'csrc/vtp_cuda.cu',
                ],
                extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']}
            )
        )
        cmdclass = {'build_ext': BuildExtension}
    else:
        cmdclass = {}
except ImportError:
    cmdclass = {}

setup(
    name='vramancer_native',
    ext_modules=ext_modules,
    cmdclass=cmdclass
)

