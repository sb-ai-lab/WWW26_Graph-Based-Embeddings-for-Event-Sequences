#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 29)) 

PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

MAX_EPOCHES=150

BATCH_SIZE=64
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
      experiment_name_without_max_epochs="barlow_twins__item_id_embed_init_with_pretrained_gnn__bs_${BATCH_SIZE}__${model_dir}__pretrain_epoches_${pretrain_epoch}__freeze_${freeze_pretrained_embs}"
      full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"
      
      echo ${full_experiment_name}

      for checkpoint_n in "${CHECKPOINTS[@]}"; do
        N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
        MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
        MODEL_PATH="models/${MODEL_NAME}.p"

        echo ""
        echo "EXPERIMENT: ${MODEL_NAME}"
        echo ""

        PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
          --config-dir conf --config-name barlow_twins_params \
          model_path=${MODEL_PATH} \
          logger_name="${experiment_name_without_max_epochs}" \
          \
          +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
          \
          trx_embedding_layers=all_except_item_id \
          +graph_path="data/graphs/weighted" \
          +trx_custom_embeddings=pretrained_graph_item_embedder \
          trx_custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
          trx_custom_embeddings.mcc_code.freeze=${freeze_pretrained_embs} \
          +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \


          # Execute the Python script for inference
          PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
              model_path=${MODEL_PATH} \
              embed_file_name="${MODEL_NAME}_embeddings" \
              inference.batch_size=400 \
              --config-dir conf --config-name barlow_twins_params

      
      done
    done
  done
done
