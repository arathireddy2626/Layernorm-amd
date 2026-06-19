#!/usr/bin/env bash
# Full benchmark suite: single-op + end-to-end, FP16/BF16/FP32 (MI300X).
set -euo pipefail
cd "$(dirname "$0")/.."

python build_ext.py

python -m src.benchmark --single-op --precision fp16
python -m src.benchmark --single-op --precision bf16
python -m src.benchmark --single-op --precision fp32

python -m src.benchmark --e2e --precision fp16
python -m src.benchmark --e2e --precision fp32
