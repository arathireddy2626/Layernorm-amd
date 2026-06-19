#!/usr/bin/env bash
# Extended sweep: additional batch sizes for single-op coverage (MI300X).
set -euo pipefail
cd "$(dirname "$0")/.."

python build_ext.py
for prec in fp16 bf16 fp32; do
  python -m src.benchmark --single-op --precision "$prec"
done
python -m src.test_correctness
