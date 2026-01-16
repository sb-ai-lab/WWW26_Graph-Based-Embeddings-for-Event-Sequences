#!/bin/bash

MAX_EPOCHES=150
DUMMY_FREEZE_URL_VALUE="false"

experiment_name_without_max_epochs=$1
checkpoint_n=$2

full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"


N_ECPOCHES_FROM_ONE=$((checkpoint_n + 1))
MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
MODEL_PATH="models/${MODEL_NAME}.p"

echo ""
echo "EXPERIMENT: ${MODEL_NAME}"
echo ""

PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
  --config-dir conf --config-name barlow_twins_params_no_url \
  model_path=${MODEL_PATH} \
  logger_name="${experiment_name_without_max_epochs}" \
  \
  +trx_custom_embeddings=ptls_id_to_llm_embedding \
  trx_custom_embeddings.url_host_preprocesed.freeze=${DUMMY_FREEZE_URL_VALUE} \
  +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
  \
  +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \

PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
  --config-dir conf --config-name barlow_twins_params_no_url \
  model_path=${MODEL_PATH} \
  embed_file_name="${MODEL_NAME}_embeddings" \
  inference.batch_size=700
