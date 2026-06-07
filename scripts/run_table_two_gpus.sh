#!/usr/bin/env bash
set -euo pipefail

if ps -eo pid=,args= | awk -v self="$$" '$1 != self && index($0, "bash scripts/run_table_two_gpus.sh") { found = 1 } END { exit found ? 0 : 1 }'; then
  echo "Another scripts/run_table_two_gpus.sh process is already active; refusing to reset the reproduction queue." >&2
  exit 3
fi

read -r -a GPUS <<<"${GPU_LIST:-0 1}"
if [ "${#GPUS[@]}" -ne 2 ]; then
  echo "GPU_LIST must contain exactly two GPU ids, for example: GPU_LIST=\"0 1\" $0" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -n "${DIGER_ENV_BIN:-}" ]; then
  if [ ! -x "${DIGER_ENV_BIN}/python" ]; then
    echo "DIGER_ENV_BIN is set but ${DIGER_ENV_BIN}/python is not executable" >&2
    exit 2
  fi
  PYTHON_BIN="${DIGER_ENV_BIN}/python"
fi

TASKS=(
  "yelp both"
  "yelp sdud"
  "yelp frqud"
  "beauty sdud"
  "beauty both"
  "instruments frqud"
  "beauty frqud"
  "instruments both"
  "instruments sdud"
)

mkdir -p reproduction_logs
QUEUE_STATE="${QUEUE_STATE:-reproduction_logs/table_queue.state}"
QUEUE_LOCK="${QUEUE_LOCK:-reproduction_logs/table_queue.lock}"
if [ ! -f "${QUEUE_STATE}" ] || [ "${RESUME_QUEUE:-0}" != "1" ]; then
  printf '0\n' > "${QUEUE_STATE}"
fi

next_task() {
  local task_index
  {
    flock -x 200
    task_index="$(cat "${QUEUE_STATE}")"
    if [ "${task_index}" -ge "${#TASKS[@]}" ]; then
      printf '\n'
    else
      printf '%s\n' "$((task_index + 1))" > "${QUEUE_STATE}"
      printf '%s\n' "${TASKS[$task_index]}"
    fi
  } 200>"${QUEUE_LOCK}"
}

run_worker() {
  local gpu="$1"
  local task dataset variant
  while true; do
    task="$(next_task)"
    if [ -z "${task}" ]; then
      echo "[$(date '+%F %T')] GPU ${gpu}: no more tasks"
      break
    fi
    read -r dataset variant <<<"${task}"
    echo "[$(date '+%F %T')] GPU ${gpu}: starting ${dataset}:${variant}"
    GPU="${gpu}" bash scripts/run_experiment.sh "${dataset}" "${variant}"
    echo "[$(date '+%F %T')] GPU ${gpu}: finished ${dataset}:${variant}"
  done
}

run_worker "${GPUS[0]}" &
pid0=$!
run_worker "${GPUS[1]}" &
pid1=$!

status=0
if ! wait "${pid0}"; then
  status=1
fi
if ! wait "${pid1}"; then
  status=1
fi

if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

"${PYTHON_BIN}" scripts/verify_results.py
