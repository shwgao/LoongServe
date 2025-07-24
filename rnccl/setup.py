from setuptools import setup
from torch.utils import cpp_extension
import os, sys

__version__ = "0.0.1"

CUDA_HOME = os.getenv("CUDA_HOME", None)
if CUDA_HOME is None:
    print("Error: CUDA_HOME not set")
    sys.exit(1)

# NCCL 路径优先用 NCCL_ROOT，否则用 Polaris 默认路径
NCCL_ROOT = "/soft/libraries/nccl/nccl_2.21.5-1+cuda12.4_x86_64"

# include 路径
include_dirs = [
    os.path.join(CUDA_HOME, "include"),
    os.path.join(NCCL_ROOT, "include"),
]

# lib 路径
library_dirs = [
    os.path.join(NCCL_ROOT, "lib"),
]

# 你可以根据需要添加 CONDA_PREFIX/lib
CONDA_PREFIX = os.getenv("CONDA_PREFIX", None)
if CONDA_PREFIX is not None:
    library_dirs.append(os.path.join(CONDA_PREFIX, "lib"))

ext_modules = [
    cpp_extension.CppExtension(
        "rnccl",
        ["src/main.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args=[],
        extra_link_args=["-lnccl"],  # 显式链接 NCCL
    ),
]

setup(
    name="RNCCL",
    version=__version__,
    author="Bingyang Wu, Shengyu Liu",
    author_email="",
    url="",
    description="Raw NCCL Bindings for Python",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.7",
)
