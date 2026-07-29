#!/usr/bin/env bash
set -eux

cd "$(dirname "$0")"/..

# --------- Environment variables are injected by ModelArts ----------
export NPROC_PER_NODE=${MA_NUM_GPUS}
export NNODES=${MA_NUM_HOSTS}
export RANK=${VC_TASK_INDEX}
export MASTER_ADDR=$(echo $VC_WORKER_HOSTS | cut -d',' -f1)
export MASTER_PORT=29500            

# set ascend BACKEND
export TORCH_DISTRIBUTED_BACKEND=hccl
export HCCL_CONNECT_TIMEOUT=7200    # Ascend Recommended Values


# ---------------------------------------------

# script & config path
TRAIN_PY=/workspace/Codes/LLaMA-Factory/src/train.py
SFT_CFG=/workspace/Codes/RoboGPT-R1/SFT/full_sft.yaml   

torchrun \
  --nproc_per_node $NPROC_PER_NODE \
  --nnodes $NNODES \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  $TRAIN_PY \
  $SFT_CFG