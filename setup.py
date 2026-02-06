import glob
import os
import platform
import shutil
import sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_cuda_home():
    """Auto-detect CUDA installation directory."""
    # 1. CUDA_HOME / CUDA_PATH env var
    for var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(var)
        if cuda_home and os.path.isdir(cuda_home):
            return cuda_home

    # 2. nvcc on PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))

    # 3. Common install paths (prefer latest version)
    for path in sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        if os.path.isdir(os.path.join(path, "include")):
            return path

    if os.path.isdir("/usr/local/cuda"):
        return "/usr/local/cuda"

    print(
        "ERROR: CUDA not found.\n"
        "  Set CUDA_HOME to your CUDA toolkit directory, e.g.:\n"
        "    export CUDA_HOME=/usr/local/cuda-12.8\n",
        file=sys.stderr,
    )
    sys.exit(1)


def find_cufile(cuda_home):
    """Locate cuFile (GPUDirect Storage) headers and library."""
    # Check for cufile.h
    header_dirs = [
        os.path.join(cuda_home, "include"),
        "/usr/include",
    ]
    header_found = any(
        os.path.isfile(os.path.join(d, "cufile.h")) for d in header_dirs
    )

    # Check for libcufile.so
    arch = platform.machine()  # x86_64, aarch64
    lib_dirs = [
        os.path.join(cuda_home, "targets", f"{arch}-linux", "lib"),
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "lib"),
        "/usr/lib",
        f"/usr/lib/{arch}-linux-gnu",
    ]
    lib_dir = None
    for d in lib_dirs:
        if any(
            os.path.isfile(os.path.join(d, f))
            for f in ("libcufile.so", "libcufile.so.0")
        ):
            lib_dir = d
            break

    if not header_found or not lib_dir:
        missing = []
        if not header_found:
            missing.append("cufile.h (header)")
        if not lib_dir:
            missing.append("libcufile.so (library)")
        print(
            f"ERROR: cuFile (GPUDirect Storage) not found: {', '.join(missing)}\n"
            "  cuFile is part of the CUDA toolkit (12.x+) or GDS package.\n"
            "  Install with: sudo apt install nvidia-gds\n"
            "  Or download from: https://developer.nvidia.com/gpudirect-storage\n",
            file=sys.stderr,
        )
        sys.exit(1)

    include_dir = os.path.join(cuda_home, "include")
    return include_dir, lib_dir


cuda_home = find_cuda_home()
include_dir, lib_dir = find_cufile(cuda_home)

print(f"GdsLLM build config:")
print(f"  CUDA home:    {cuda_home}")
print(f"  Include dir:  {include_dir}")
print(f"  Library dir:  {lib_dir}")

setup(
    name="gdsllm",
    version="0.1.0",
    description="LLM inference runtime with NVMe-to-VRAM weight streaming via GPUDirect Storage",
    long_description=open("README.md").read() if os.path.isfile("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/rscunha13/gdsllm",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="gdsllm.runtime.gds_io_ext",
            sources=[
                "gdsllm/runtime/gds_io.cu",
                "gdsllm/runtime/gguf_dequant.cu",
                "gdsllm/runtime/fused_gemv.cu",
                "gdsllm/runtime/gds_bindings.cpp",
            ],
            include_dirs=[include_dir],
            library_dirs=[lib_dir],
            libraries=["cufile"],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "gdsllm=gdsllm.cli:main",
        ],
    },
)
