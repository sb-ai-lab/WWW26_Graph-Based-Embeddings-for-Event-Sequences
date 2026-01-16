#!/bin/bash

PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'
MAX_EPOCHES=150
BATCH_SIZE=64
FREEZE_PRETRAINED_EMBS_OPTIONS=(false)
CHECKPOINTS=($(seq 9 10 29))  # Example checkpoints

declare -A model_epoch_map
model_epoch_map["wl-0.5_gnn-GAT_residual-True_weight_decay-0"]="11 31 51 71 91 111 131 171" 
model_epoch_map["wl-0.5_gnn-GraphSAGE_residual-True_weight_decay-0"]="11 31 51 71 91 111 131 171" 


for freeze_pretrained_embs in "${FREEZE_PRETRAINED_EMBS_OPTIONS[@]}"; do
  for model_dir in "${!model_epoch_map[@]}"; do

    IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

    for pretrain_epoch in "${EPOCHES[@]}"; do
      f_name="${pretrain_epoch}.pt"
      embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"
      experiment_name_without_max_epochs="cpc__item_id_embed_init_with_pretrained_gnn__bs_${BATCH_SIZE}__${model_dir}__pretrain_epoches_${pretrain_epoch}__freeze_${freeze_pretrained_embs}"
      full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"

      ./get_models_from_checkpoints_single_model.sh \
        "${freeze_pretrained_embs}" \
        "${model_dir}" \
        "${pretrain_epoch}" \
        "${embeddings_path}" \
        "${experiment_name_without_max_epochs}" \
        "${full_experiment_name}" \
        "${CHECKPOINTS[*]}"  # Pass checkpoints as a space-separated string
    done
  done
done