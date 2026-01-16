#!/bin/bash

PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'
MAX_EPOCHES=150
TRAIN_BATCH_SIZE=64
FREEZE_PRETRAINED_EMBS_OPTIONS=(false)

declare -A model_epoch_map
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_64"]="15 51 101 151 191"
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_128"]="15 51 101 151 191"

for freeze_pretrained_embs in "${FREEZE_PRETRAINED_EMBS_OPTIONS[@]}"; do
  for model_dir in "${!model_epoch_map[@]}"; do
    # Get the list of epochs for the current model directory
    IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

    for pretrain_epoch in "${EPOCHES[@]}"; do
      f_name="${pretrain_epoch}.pt"
      embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"
      experiment_name="cpc__item_id_embed_init_with_pretrained_gnn__bs_${TRAIN_BATCH_SIZE}__${model_dir}__pretrain_epoches_${pretrain_epoch}__freeze_${freeze_pretrained_embs}__epoches_${MAX_EPOCHES}"

      bash \
        ./bin/scenarios_cpc/item_id_embedding_initialized_with_pretrained_gnn/run_single.sh \
        "${MAX_EPOCHES}" \
        "${TRAIN_BATCH_SIZE}" \
        "${freeze_pretrained_embs}" \
        "${embeddings_path}" \
        "${experiment_name}"
    done
  done
done
