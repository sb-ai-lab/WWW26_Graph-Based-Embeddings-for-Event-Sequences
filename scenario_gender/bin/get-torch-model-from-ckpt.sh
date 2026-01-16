#!/bin/bash

# Set the root directory for pretrained models
PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

# Set the maximum number of epochs
MAX_EPOCHES=40

epoch=50
f_name="${epoch}.pt"
model_dir="wl-0.5_gnn-GAT_res-True_wd-0.01"

embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

# Define the experiment name
experiment_name="coles_gnn__pretrained_${model_dir}__pretrain_epoches_${epoch}__epoches_${MAX_EPOCHES}"

PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
    --config-dir conf --config-name coles_gnn_pretrained_params \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/${experiment_name}.p" \
    logger_name="${experiment_name}" \
    data_module.train_batch_size=64 \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=64 \
    data_module.valid_num_workers=4 \
    pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings_path="${embeddings_path}" \
    device="cpu" \
    +ckpt_path="./checkpoints/${experiment_name}/epoch\=39.ckpt"


# ./checkpoints/coles_gnn__pretrained_wl-0.5_gnn-GAT_res-True_wd-0.01__pretrain_epoches_50__epoches_40/epoch=39.ckpt
