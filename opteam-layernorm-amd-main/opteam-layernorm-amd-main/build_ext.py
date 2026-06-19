"""Build the HIP extension using JIT compilation (hipcc via torch cpp_extension)."""
import torch.utils.cpp_extension as ext

module = ext.load(
    name="denominator_hip",
    sources=[
        "csrc/denominator.cpp",
        "csrc/denominator_kernel.hip",
    ],
    extra_cuda_cflags=["-O3", "-ffast-math", "--offload-arch=gfx942"],
    extra_cflags=["-O3"],
    verbose=True,
)

print("Build successful!")
print(f"Module: {module}")
print(f"Functions: {dir(module)}")
