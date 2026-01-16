#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 


# Set the root directory for pretrained models
PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

# Set the maximum number of epochs
MAX_EPOCHES=150

epoch=100
f_name="${epoch}.pt"
model_dir="wl-0.5_gnn-GraphSAGE_res-True_wd-0"

embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

# Define the experiment name
experiment_name="coles_gnn__pretrained_${model_dir}__pretrain_epoches_${epoch}__epoches_${MAX_EPOCHES}"


for checkpoint_n in "${CHECKPOINTS[@]}"; do

  MODEL_PATH="models/${experiment_name}__ckpt_epoch_${checkpoint_n}.p"

  PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
    --config-dir conf --config-name coles_gnn_pretrained_params \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path=${MODEL_PATH} \
    logger_name="${experiment_name}" \
    data_module.train_batch_size=64 \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=64 \
    data_module.valid_num_workers=4 \
    pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings_path="${embeddings_path}" \
    +ckpt_path="./checkpoints/${experiment_name}/epoch\=${checkpoint_n}.ckpt" \
    # device="cpu" \


  # # Execute the Python script for inference
  # PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
  #     model_path=${MODEL_PATH} \
  #     embed_file_name="${experiment_name}__ckpt_epoch_${checkpoint_n}_embeddings" \
  #     inference.batch_size=16 \
  #     --config-dir conf --config-name coles_gnn_pretrained_params

  #     # +inference.devices=0 \
done

