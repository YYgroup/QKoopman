#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

export NCCL_DEBUG=WARN
export DS_AUTOTUNING_LOG_LEVEL=INFO

export NCCL_ALGO=Tree,Ring
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_LEVEL=NVL
export NCCL_NET_SHARED_COMMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=12
export NCCL_NVLS_ENABLE=1

source ./miniconda3/bin/activate
source activate zby
cd ./shear
export PYTHONUNBUFFERED=1

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

deepspeed --num_gpus ${NUM_GPUS} \
          *.py  \
          --deepspeed \
          --deepspeed_config ds_config.json \
          > run.log 2>&1