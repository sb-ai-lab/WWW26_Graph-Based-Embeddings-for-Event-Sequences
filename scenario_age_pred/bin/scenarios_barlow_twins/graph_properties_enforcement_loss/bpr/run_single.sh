#!/bin/bash


LOSS_ALPHA=$1
N_TRIPLETS_PER_ANCHOR_USER=$2
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY=$3

MAX_EPOCHES=150
TRAIN_BATCH_SIZE=64

MODEL_NAME="barlow_twins_bpr__batch_size_${TRAIN_BATCH_SIZE}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}__epoches_${MAX_EPOCHES}"

model_path="models/${MODEL_NAME}_model.p"

PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
  --config-dir conf --config-name barlow_twins_params \
  \
  data_module.train_data._target_="ptls_extension_2024_research.ColesDataset" \
  +data_module.train_data.col_client_id="encoded_client_id" \
  data_module.valid_data._target_="ptls_extension_2024_research.ColesDataset" \
  +data_module.valid_data.col_client_id="encoded_client_id" \
  \
  ++data_module.train_drop_last=true \
  \
  model_path="${model_path}" \
  logger_name="${MODEL_NAME}"  \
  \
  ++graph_path="./data/graphs/weighted" \
  \
  data_module.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data_module.train_num_workers=4 \
  data_module.valid_batch_size=64 \
  data_module.valid_num_workers=4  \
  \
  +loss=contrastive_loss_and_additional_loss_convex_combination \
  loss.alpha=${LOSS_ALPHA} \
  loss.loss2.triplet_selector.num_triplets_per_anchor_user=${N_TRIPLETS_PER_ANCHOR_USER} \
  loss.loss2.triplet_selector.min_elements_in_bin=${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY} \
  loss/loss1=barlow \
  pl_module.loss=\${loss} \
  \
  +trainer.checkpoints_every_n_val_epochs=10 \
  +trainer.checkpoint_dirpath="./checkpoints/${MODEL_NAME}" \
  +trainer.checkpoint_filename="\{epoch\}" \
  trainer.max_epochs="${MAX_EPOCHES}" \
  \
  hydra.run.dir="hydra_outputs/${MODEL_NAME}" \
  \
  +additional_artifacts_to_save="[git_commit_hash]" \
  # device="cpu"


# Extract embeddings
PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
  --config-dir conf --config-name barlow_twins_params \
  model_path="${MODEL_PATH}" \
  embed_file_name="${MODEL_NAME}_embeddings" \
  inference.batch_size=1700 \
  # +inference.devices=0 \
