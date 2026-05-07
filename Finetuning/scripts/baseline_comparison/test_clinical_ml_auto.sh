#!/usr/bin/env bash
set -euo pipefail
MODEL_TYPE="${1:-lr}"
bash Finetuning/scripts/baseline_comparison/run_clinical_ml.sh "${MODEL_TYPE}"
