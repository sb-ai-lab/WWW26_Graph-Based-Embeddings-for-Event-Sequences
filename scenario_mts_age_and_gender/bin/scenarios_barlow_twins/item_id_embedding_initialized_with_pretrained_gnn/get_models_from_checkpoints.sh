#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 

BATCH_SIZE=40
PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'
MAX_EPOCHES=150
FREEZE_PRETRAINED_EMBS_OPTIONS=(false)

declare -A model_epoch_map
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0--new"]="11" 


for freeze_pretrained_embs in "${FREEZE_PRETRAINED_EMBS_OPTIONS[@]}"; do
  for model_dir in "${!model_epoch_map[@]}"; do
    # Get the list of epochs for the current model directory
    IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"
    for pretrain_epoch in "${EPOCHES[@]}"; do

      experiment_name_without_max_epochs="barlow_twins__item_id_embed_init_with_pretrained_gnn__bs_${BATCH_SIZE}__${model_dir}__pretrain_epoches_${pretrain_epoch}__freeze_${freeze_pretrained_embs}"
      full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"

      f_name="${pretrain_epoch}.pt"
      embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

      for checkpoint_n in "${CHECKPOINTS[@]}"; do
      ./bin/scenarios_barlow_twins/item_id_embedding_initialized_with_pretrained_gnn/get_models_from_checkpoints_single_model.sh \
        "${freeze_url}" \
        "${experiment_name_without_max_epochs}" \
        "${full_experiment_name}" \
        "${embeddings_path}" \
        "${checkpoint_n}"
      done
    done
  done
done
