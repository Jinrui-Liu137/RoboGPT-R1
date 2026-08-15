#!/usr/bin/env bash

# Run RFT with the EasyR1 environment activated.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export PYTHONUNBUFFERED=1
export DEBUG_MODE=false

# workspace & experiments setting
ROBOGPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EASYR1_ROOT="$(cd "${ROBOGPT_ROOT}/../EasyR1" && pwd)"
EXPERIMENT_ROOT="/workspace/Experiments/RoboGPT-R1"
EXP_NAME="${EXP_NAME:-exp1--sft+rft-lcs_reward}"
EXPERIMENT_DIR="${EXPERIMENT_ROOT}/${EXP_NAME}"


RFT_CONFIG="${ROBOGPT_ROOT}/RFT/rft_no_example.yaml"
MODEL_PATH="${EXPERIMENT_ROOT}/exp0--only_sft/weight1"
DATA_DIR="${ROBOGPT_ROOT}/dataset_PARQUET"


if [[ ! -f "${RFT_CONFIG}" ]]; then
    echo "RFT configuration not found: ${RFT_CONFIG}" >&2
    exit 1
fi

if [[ ! -d "${EASYR1_ROOT}" ]]; then
    echo "EasyR1 repository not found: ${EASYR1_ROOT}" >&2
    exit 1
fi

mkdir -p "${EXPERIMENT_DIR}"
export REWARD_LOG_PATH="${EXPERIMENT_DIR}/rft_lcs_reward.log"

cd "${EASYR1_ROOT}"
python -m verl.trainer.main \
    config=${RFT_CONFIG} \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.save_checkpoint_path="${EXPERIMENT_DIR}/weight1" \
    worker.rollout.tensor_parallel_size=1 \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=${NPROC_PER_NODE}

