#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

seeds=($(python3 ${SCRIPT_DIR}/load_settings.py seeds))
num_parallel=${#seeds[@]}

filename=$(basename "$0")
EXP_NAME="${filename%.*}"

echo "Starting ${EXP_NAME} with ${num_parallel} parallel processes"

for i in $(seq 0 $((num_parallel-1)))
do
    common_args=$(python3 ${SCRIPT_DIR}/load_settings.py args ${seeds[$i]})
    
    CUDA_VISIBLE_DEVICES=$i \
    nohup python3 ${REPO_ROOT}/pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=${EXP_NAME} \
    --only_bc_loss=True \
    --free_level=-10000.0 \
    ${common_args} \
    > ${EXP_NAME}_seed${seeds[$i]}.log 2>&1 &
done

echo "All processes started. Check logs: ${EXP_NAME}_seed*.log"
