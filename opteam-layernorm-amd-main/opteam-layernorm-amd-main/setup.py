from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# On ROCm builds of PyTorch, CUDAExtension compiles with hipcc; the "nvcc"
# compile-arg key is forwarded to hipcc.
setup(
    name="denominator_hip",
    ext_modules=[
        CUDAExtension(
            name="denominator_hip",
            sources=[
                "csrc/denominator.cpp",
                "csrc/denominator_kernel.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-ffast-math", "--offload-arch=gfx942"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
