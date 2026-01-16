#!/bin/bash

freeze_url=$1
experiment_name_without_max_epochs=$2
full_experiment_name=$3
checkpoint_n=$4


N_ECPOCHES_FROM_ONE=$((checkpoint_n + 1))
MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
MODEL_PATH="models/${MODEL_NAME}.p"

PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
  --config-dir conf --config-name barlow_twins_params_no_url \
  model_path=${MODEL_PATH} \
  logger_name="${full_experiment_name}" \
  \
  +trx_custom_embeddings=ptls_id_to_llm_embedding \
  trx_custom_embeddings.url_host_preprocesed.freeze=${freeze_url} \
  +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
  \
  +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt"

# Compute embeddings
PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
  --config-dir conf --config-name barlow_twins_params_no_url \
  model_path=${MODEL_PATH} \
  embed_file_name="${MODEL_NAME}_embeddings" \
  inference.batch_size=800