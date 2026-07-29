#!/usr/bin/env bash
set -eux

cd "$(dirname "$0")"/..

# --------- Distributed/Device Parameters (Modify as Needed) ----------
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8           
export NPROC_PER_NODE=8                      # Consistent with the number of devices shown above
export NNODES=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30000

# set nvidia BACKEND
export TORCH_DISTRIBUTED_BACKEND=nccl

# ------------------------------------------------

# script & config path
TRAIN_PY=/workspace/Codes/LLaMA-Factory/src/train.py
SFT_CFG=/workspace/Codes/RoboGPT-R1/SFT/full_sft.yaml             

# start training
torchrun \
  --nproc_per_node "$NPROC_PER_NODE" \
  --nnodes "$NNODES" \
  --node_rank "$RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$TRAIN_PY" "$SFT_CFG"
