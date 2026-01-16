#!/bin/bash

LOSS_ALPHA=$1
N_TRIPLETS_PER_ANCHOR_USER=$2
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY=$3
checkpoints_string=$4

MAX_EPOCHES=150
TRAIN_BATCH_SIZE=64

experiment_name_without_max_epochs="barlow_twins_bpr__batch_size_${TRAIN_BATCH_SIZE}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}"
full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"

IFS=' ' read -r -a CHECKPOINTS <<< "$checkpoints_string"

for checkpoint_n in "${CHECKPOINTS[@]}"; do
  N_ECPOCHES_FROM_ONE=$((checkpoint_n + 1))
  MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
  MODEL_PATH="models/${MODEL_NAME}.p"

  echo ""
  echo "EXPERIMENT: ${MODEL_NAME}"
  echo ""

  PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
    --config-dir conf --config-name barlow_twins_params \
    model_path=${MODEL_PATH} \
    logger_name="${experiment_name_without_max_epochs}" \
    +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \

  PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
    --config-dir conf --config-name barlow_twins_params \
    model_path=${MODEL_PATH} \
    embed_file_name="${MODEL_NAME}_embeddings" \
    inference.batch_size=1700
done