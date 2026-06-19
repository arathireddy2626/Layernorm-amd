"""Load the HIP denominator extension, building via JIT (hipcc) if needed."""
import os
import torch.utils.cpp_extension as ext

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# On ROCm builds of PyTorch, cpp_extension drives hipcc; the "cuda" cflags
# are forwarded to hipcc.
denominator_hip = ext.load(
    name="denominator_hip",
    sources=[
        os.path.join(_ROOT, "csrc", "denominator.cpp"),
        os.path.join(_ROOT, "csrc", "denominator_kernel.hip"),
    ],
    extra_cuda_cflags=["-O3", "-ffast-math", "--offload-arch=gfx942"],
    extra_cflags=["-O3"],
    verbose=False,
)
