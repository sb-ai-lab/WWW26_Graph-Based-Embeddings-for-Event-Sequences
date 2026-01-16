#!/bin/bash

freeze_pretrained_embs=$1
model_dir=$2
pretrain_epoch=$3
embeddings_path=$4
experiment_name_without_max_epochs=$5
full_experiment_name=$6
checkpoints_string=$7

IFS=' ' read -r -a CHECKPOINTS <<< "$checkpoints_string"

for checkpoint_n in "${CHECKPOINTS[@]}"; do
  N_ECPOCHES_FROM_ONE=$((checkpoint_n + 1))
  MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
  MODEL_PATH="models/${MODEL_NAME}.p"

  echo ""
  echo "EXPERIMENT: ${MODEL_NAME}"
  echo ""

  PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
    --config-dir conf --config-name barlow_twins_params \
    model_path=${MODEL_PATH} \
    logger_name="${experiment_name_without_max_epochs}" \
    +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
    trx_embedding_layers=all_except_item_id \
    ++graph_path="data/graphs/weighted" \
    +trx_custom_embeddings=pretrained_graph_item_embedder \
    trx_custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
    trx_custom_embeddings.mcc_code.freeze=${freeze_pretrained_embs} \
    +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings}

  PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
    --config-dir conf --config-name barlow_twins_params \
    model_path=${MODEL_PATH} \
    embed_file_name="${MODEL_NAME}_embeddings" \
    inference.batch_size=1700
done