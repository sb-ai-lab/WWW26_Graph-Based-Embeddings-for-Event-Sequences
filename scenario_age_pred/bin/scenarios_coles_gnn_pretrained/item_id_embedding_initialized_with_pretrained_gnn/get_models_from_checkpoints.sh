#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 89)) 


PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

MAX_EPOCHES=150

# * Batch size 64 is over 2 Gb
# * Batch size 48 is bad because last batch is too small.
#   and we cannot sample 5 negative examples (neg_count in HardNegativeMining)
# * Batch size 46 is sometimes ok for 2Gb gpu, but once CUDA went out of memory. It was just 1 Mb short
BATCH_SIZE=40
SPLIT_COUNT=2


# Declare an associative array to map model directories to epoch lists
declare -A model_epoch_map

model_epoch_map["wl-0.5_gnn-GAT_residual-True_weight_decay-0"]="31 91" 
model_epoch_map["wl-0.5_gnn-GraphSAGE_residual-True_weight_decay-0"]="31 71" 
model_epoch_map["GRACE-gnn-GAT-weight_decay-0.00001__residual-true__num_layers_3__emb_size_128"]="11"



for model_dir in "${!model_epoch_map[@]}"; do
  # Get the list of epochs for the current model directory
  IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

  for pretrain_epoch in "${EPOCHES[@]}"; do

    f_name="${pretrain_epoch}.pt"
    embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

    experiment_name_without_max_epochs="coles_only__item_id_embed_init_with_pretrained_gnn__bs_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__${model_dir}__pretrain_epoches_${pretrain_epoch}"
    full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"
    
    for checkpoint_n in "${CHECKPOINTS[@]}"; do
      N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
      MODEL_NAME="${experiment_name_without_max_epochs}__${N_ECPOCHES_FROM_ONE}_epoches"
      MODEL_PATH="models/${MODEL_NAME}.p"

      PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
        --config-dir conf --config-name mles_params \
        data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
        data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
        pl_module.validation_metric.K=1 \
        pl_module.lr_scheduler_partial.step_size=60 \
        \
        model_path=${MODEL_PATH} \
        logger_name="${experiment_name_without_max_epochs}" \
        \
        data_module.train_batch_size=${BATCH_SIZE} \
        data_module.train_num_workers=4 \
        data_module.valid_batch_size=64 \
        data_module.valid_num_workers=4 \
        \
        +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
        \
        trx_embedding_layers=all_except_item_id \
        +trx_custom_embeddings=pretrained_graph_item_embedder \
        trx_custom_embeddings.small_group.embeddings.f="${embeddings_path}" \
        +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
        # device="cpu" \


      # Execute the Python script for inference
      PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
          model_path=${MODEL_PATH} \
          embed_file_name="${MODEL_NAME}_embeddings" \
          inference.batch_size=64 \
          --config-dir conf --config-name mles_params

          # +inference.devices=0 \

    done
  done
done
