from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gdsllm",
    version="0.1.0",
    description="LLM inference runtime with NVMe-to-VRAM weight streaming via GPUDirect Storage",
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
            include_dirs=["/usr/local/cuda-12.8/include"],
            library_dirs=["/usr/local/cuda-12.8/targets/x86_64-linux/lib"],
            libraries=["cufile"],
            extra_compile_args={
                "cxx": ["-std=c++17"],
                "nvcc": ["-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
