#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 


MAX_EPOCHES=150

# * Batch size 64 is over 2 Gb
# * Batch size 48 is bad because last batch is too small.
#   and we cannot sample 5 negative examples (neg_count in HardNegativeMining)
# * Batch size 46 is sometimes ok for 2Gb gpu, but once CUDA went out of memory. It was just 1 Mb short
BATCH_SIZE=40
SPLIT_COUNT=2


EXPERIMENT_NAME_WITHOUT_MAX_EPOCHS="coles__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}"
FULL_EXPERIMNT_NAME="${EXPERIMENT_NAME_WITHOUT_MAX_EPOCHS}__${MAX_EPOCHES}_epoches"

for checkpoint_n in "${CHECKPOINTS[@]}"; do
  N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
  MODEL_NAME="${EXPERIMENT_NAME_WITHOUT_MAX_EPOCHS}__${N_ECPOCHES_FROM_ONE}_epoches"
  MODEL_PATH="models/${MODEL_NAME}.p"

  PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
    --config-dir conf --config-name mles_params \
    data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
    data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path=${MODEL_PATH} \
    logger_name="${EXPERIMENT_NAME_WITHOUT_MAX_EPOCHS}" \
    data_module.train_batch_size=${BATCH_SIZE} \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=64 \
    data_module.valid_num_workers=4 \
    +ckpt_path="./checkpoints/${FULL_EXPERIMNT_NAME}/epoch\=${checkpoint_n}.ckpt" \
    # device="cpu" \


  # Execute the Python script for inference
  PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
      model_path=${MODEL_PATH} \
      embed_file_name="${MODEL_NAME}_embeddings" \
      inference.batch_size=96 \
      --config-dir conf --config-name mles_params

      # +inference.devices=0 \
done

