# Fused LayerNorm+Linear — AMD MI300X (HIP/ROCm)

HIP port of the H100 CUDA fused LayerNorm+Linear kernels
([`h100-fused-layernorm-linear`](https://github.com/Runaraai/h100-fused-layernorm-linear))
for AMD Instinct MI300X (gfx942), built with `hipcc` via the PyTorch C++ extension toolchain.

## Method

Per formula (7) in the task description (`cuda-task.txt`), `LayerNorm(x) @ F` is rewritten as

```
x @ F_new / v(x) + b_new        F_new = (I - E/h) @ diag(gamma) @ F
```

so the LayerNorm collapses into the Linear weights, leaving only the per-row denominator
`v(x) = ||x - mean(x)||_2` to compute — which is then fused with the normalize step.

## Kernel variants

| Variant | Strategy |
|---|---|
| V0 | Two-pass denominator (separate kernel) |
| V1 | Fused denominator + normalize, 256 threads |
| V2 | Welford single-pass denominator |
| V3 | Welford + fused normalize, 512 threads |
| RMSNorm V1 / V3 | RMSNorm-fused equivalents |

FP32, FP16 and BF16 are supported in all fused paths.

## MI300X (wave64) port notes

- `WARP_SIZE = 64` (gfx942 wavefront); block reductions use 4 waves of 64 lanes
  instead of 8 warps of 32.
- `__shfl_down_sync(mask, ...)` → `__shfl_down(..., WARP_SIZE)` (HIP has no
  `_sync` shuffle variants; width is passed explicitly).
- Vectorized `float4` / `half2` loads carry over unchanged.
- Streams via `at::hip::getCurrentHIPStreamMasqueradingAsCUDA()`.

## Build & run

```bash
python build_ext.py                  # JIT build (hipcc, ROCm >= 6.x)
python -m src.test_correctness       # parity vs eager LayerNorm+Linear
scripts/run_full_benchmark.sh        # single-op + end-to-end benchmarks
```

## Results

See [RESULTS.md](RESULTS.md) for the MI300X measurement summary.
