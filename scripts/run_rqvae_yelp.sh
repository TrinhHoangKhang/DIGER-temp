#!/usr/bin/env bash
set -euo pipefail

GPU="${RQVAE_GPU:-${GPU:-0}}"
RQVAE_GPU="${GPU}" bash scripts/run_rqvae_pretrain.sh yelp
