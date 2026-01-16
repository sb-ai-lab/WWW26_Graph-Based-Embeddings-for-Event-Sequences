#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 

PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

MAX_EPOCHES=150

BATCH_SIZE=40
SPLIT_COUNT=2
FREEZE_PRETRAINED_EMBS_OPTIONS=(true false)


FREEZE_URL_OPTIONS=("true" "false")


declare -A model_epoch_map

# FILL ME!!!
model_epoch_map["sample_model"]="11 22"
# model_epoch_map["model_name2"]="11 22"

for freeze_pretrained_embs in "${FREEZE_PRETRAINED_EMBS_OPTIONS[@]}"; do
  for model_dir in "${!model_epoch_map[@]}"; do
    # Get the list of epochs for the current model directory
    IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

    for pretrain_epoch in "${EPOCHES[@]}"; do

      f_name="${pretrain_epoch}.pt"
      embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"
      experiment_name_without_max_epochs="coles_item_id_embed_init_with_pretrained_gnn__bs_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__${model_dir}__pretrain_epoches_${pretrain_epoch}__freeze_${freeze_pretrained_embs}"
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
          --config-dir conf --config-name mles_params_no_url \
          data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
          data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
          pl_module.validation_metric.K=1 \
          pl_module.lr_scheduler_partial.step_size=60 \
          \
          model_path=${MODEL_PATH} \
          logger_name="${experiment_name_without_max_epochs}" \
          \
          +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
          \
          trx_embedding_layers=all_except_item_id \
          +graph_path="data/graphs/weighted_train_graph_has_test_items" \
          +trx_custom_embeddings=pretrained_graph_item_embedder \
          trx_custom_embeddings.url_host_preprocesed.embeddings.f="${embeddings_path}" \
          trx_custom_embeddings.url_host_preprocesed.freeze=${freeze_pretrained_embs} \
          +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \


          # Execute the Python script for inference
          PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
              model_path=${MODEL_PATH} \
              embed_file_name="${MODEL_NAME}_embeddings" \
              inference.batch_size=400 \
              --config-dir conf --config-name mles_params_no_url

      
      done
    done
  done
done
