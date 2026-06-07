#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <beauty|instruments|yelp> <frqud|sdud|both>" >&2
  exit 2
fi

DATASET="$1"
VARIANT="$2"
GPU="${GPU:-0}"
ACCEL_CFG="${ACCEL_CFG:-accelerate_config.yaml}"
CONFIG="config/${DATASET}_jo.yaml"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

if [ -n "${DIGER_ENV_BIN:-}" ]; then
  if [ ! -x "${DIGER_ENV_BIN}/accelerate" ]; then
    echo "DIGER_ENV_BIN is set but ${DIGER_ENV_BIN}/accelerate is not executable" >&2
    exit 2
  fi
  ACCELERATE_BIN="${DIGER_ENV_BIN}/accelerate"
fi

if [ ! -f "${CONFIG}" ]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi

COMMON_ARGS=(
  --config "${CONFIG}"
  --rqvae_path="./rqvae_ckpt/${DATASET}/best_collision_model.pth"
  --lr_rec=0.001
  --lr_id=0.00001
  --weight_decay=0.05
  --freeze_semantic_embedding=true
  --freeze_id_encoder=true
  --freeze_id_encoder_layers=0
  --freeze_id_decoder=true
  --freeze_id_epochs=0
  --freeze_rq=false
  --stop_gumbel_sampling_epoch=0
  --code_loss_weight=1.0
  --recon_loss_weight=1.0
  --vq_loss_weight=1.0
  --qs_loss_weight=0
  --use_soft_frequency=false
  --use_gate_network=false
)

case "${DATASET}" in
  beauty)
    COMMON_ARGS+=(--epochs=120 --early_stop=15 --eval_batch_size=32 --num_beams=20 --gumbel_tau=2)
    ;;
  instruments)
    COMMON_ARGS+=(--epochs=100 --early_stop=15 --eval_batch_size=32 --num_beams=20 --gumbel_tau=2)
    ;;
  yelp)
    COMMON_ARGS+=(--epochs=200 --early_stop=20 --eval_batch_size=16 --num_beams=80 --gumbel_tau=1.5)
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

case "${DATASET}:${VARIANT}" in
  beauty:frqud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.5 --usage_momentum=0.99 --use_learnable_sigma_gumbel=false --gumbel_hard_switch_epoch=0)
    ;;
  beauty:sdud)
    RUN_ARGS=(--use_adaptive_selection=false --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.7 --gumbel_hard_switch_epoch=0)
    ;;
  beauty:both)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.5 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --lr_sigma=0.001 --sigma_reg_weight=2.0 --initial_std=1.0 --noise_type=gumbel --use_cosine_annealing=false --use_dynamic_sigma_lr=true --gate_loss_weight=0 --gumbel_hard_switch_epoch=0)
    ;;
  instruments:frqud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=2.0 --usage_momentum=0.99 --use_learnable_sigma_gumbel=false --gumbel_hard_switch_epoch=0)
    ;;
  instruments:sdud)
    RUN_ARGS=(--use_adaptive_selection=false --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=1.0 --noise_type=gumbel --sigma_lambda=1.8 --gumbel_hard_switch_epoch=0)
    ;;
  instruments:both)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.5 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=1.5 --noise_type=gumbel --sigma_lambda=1.8 --gumbel_hard_switch_epoch=0)
    ;;
  yelp:frqud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.1 --usage_momentum=0.99 --use_learnable_sigma_gumbel=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.0)
    ;;
  yelp:sdud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=false --hot_threshold_ratio=1.1 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_simple_uncertainty_loss=true --use_plain_code_loss=false --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.7)
    ;;
  yelp:both)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.1 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_simple_uncertainty_loss=true --use_plain_code_loss=false --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.0)
    ;;
  *)
    echo "Unknown variant: ${DATASET}:${VARIANT}" >&2
    exit 2
    ;;
esac

mkdir -p reproduction_logs
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_LOG="reproduction_logs/${DATASET}_${VARIANT}_${STAMP}.log"

echo "Running ${DATASET}:${VARIANT} on GPU ${GPU}"
echo "stdout log: ${OUT_LOG}"
echo "accelerate: ${ACCELERATE_BIN}"

CUDA_VISIBLE_DEVICES="${GPU}" "${ACCELERATE_BIN}" launch --config_file "${ACCEL_CFG}" main.py "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}" 2>&1 | tee "${OUT_LOG}"
