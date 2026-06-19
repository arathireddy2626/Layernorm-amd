# MI300X Benchmark Summary

**Hardware:** AMD Instinct MI300X 192GB HBM3 (gfx942) · ROCm 7.0
**Campaign:** 2026-03-10 · single-op fused LayerNorm+Linear vs unfused eager baseline
**Precision:** FP16 (attention projection shapes); FP32/BF16 covered in the variant sweep

| Model shape | Batch size | Op | Speedup vs unfused |
|---|---|---|---|
| OPT-6.7b (h=4096) | 1 | attn projection | **1.79x** |
| OPT-6.7b (h=4096) | 32 | attn projection | **1.33x** |
| OPT-1.3b (h=2048) | 512 | attn projection | **1.24x** |

Kernel variants V0 / V1 / V3 were benchmarked for both LayerNorm and RMSNorm
fusions across FP32 / FP16 / BF16; the table above reports the best fused variant
per configuration.

> Raw benchmark JSON archives from the March campaign reside on the MI300X
> benchmark host; this summary mirrors the delivered status report.
