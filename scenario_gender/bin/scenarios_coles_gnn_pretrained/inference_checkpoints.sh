#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 99 -10 9)) 
# CHECKPOINTS=($(seq 9 10 149)) 
# CHECKPOINTS=($(seq 149 -10 9)) 


# Set the root directory for pretrained models
PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

# Set the maximum number of epochs
MAX_EPOCHES=150

pretrain_epoch=100
model_dir="wl-0.5_gnn-GraphSAGE_res-True_wd-0"

# Define the experiment name
experiment_name="coles_gnn__pretrained_${model_dir}__pretrain_epoches_${pretrain_epoch}__epoches_${MAX_EPOCHES}"


for checkpoint_n in "${CHECKPOINTS[@]}"; do

  MODEL_PATH="models/${experiment_name}__ckpt_epoch_${checkpoint_n}.p"

  # Execute the Python script for inference
  PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
      model_path=${MODEL_PATH} \
      embed_file_name="${experiment_name}__ckpt_epoch_${checkpoint_n}_embeddings" \
      inference.batch_size=16 \
      --config-dir conf --config-name coles_gnn_pretrained_params
      # +inference.devices=0 \
done

