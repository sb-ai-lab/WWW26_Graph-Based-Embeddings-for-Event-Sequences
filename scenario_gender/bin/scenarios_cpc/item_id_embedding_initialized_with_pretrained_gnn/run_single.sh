#!/bin/bash

MAX_EPOCHES=$1
TRAIN_BATCH_SIZE=$2
freeze_pretrained_embs=$3
embeddings_path=$4
experiment_name=$5


PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
  --config-dir conf --config-name cpc_params \
  model_path="models/${experiment_name}.p" \
  logger_name="${experiment_name}" \
  \
  data_module.train_batch_size=${TRAIN_BATCH_SIZE} \
  data_module.train_num_workers=4 \
  data_module.valid_batch_size=64 \
  data_module.valid_num_workers=4 \
  \
  +trainer.checkpoints_every_n_val_epochs=10 \
  +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
  +trainer.checkpoint_filename="\{epoch\}" \
  trainer.max_epochs=${MAX_EPOCHES} \
  \
  hydra.run.dir="hydra_outputs/${experiment_name}" \
  \
  trx_embedding_layers=all_except_item_id__cpc_default \
  +graph_path="data/graphs/weighted" \
  +trx_custom_embeddings=pretrained_graph_item_embedder \
  trx_custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
  trx_custom_embeddings.mcc_code.freeze=${freeze_pretrained_embs} \
  +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
  \
  +additional_artifacts_to_save="[git_commit_hash]" \
  # device="cpu"


# Extract embeddings
PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
  model_path="models/${experiment_name}.p" \
  embed_file_name="${experiment_name}_embeddings" \
  inference.batch_size=128 \
  --config-dir conf --config-name cpc_params
  # +inference.devices=0 \
