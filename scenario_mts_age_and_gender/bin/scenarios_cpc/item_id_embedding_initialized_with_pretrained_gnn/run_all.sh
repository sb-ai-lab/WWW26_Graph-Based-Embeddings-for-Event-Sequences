#!/bin/bash

PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'
FREEZE_PRETRAINED_EMBS_OPTIONS=(false)


declare -A model_epoch_map
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0--new"]="31" 


for freeze_pretrained_embs in "${FREEZE_PRETRAINED_EMBS_OPTIONS[@]}"; do
  for model_dir in "${!model_epoch_map[@]}"; do
    # Get the list of epochs for the current model directory
    IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"
    for pretrain_epoch in "${EPOCHES[@]}"; do
        bash ./run_unfreeze_parametrized \
          "${freeze_pretrained_embs}" \
          "${model_dir}" \
          "${pretrain_epoch}"
    done
  done
done
