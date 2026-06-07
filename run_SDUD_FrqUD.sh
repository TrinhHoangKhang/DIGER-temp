#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-beauty}"
GPU="${GPU:-0}" exec bash scripts/run_experiment.sh "${DATASET}" both
