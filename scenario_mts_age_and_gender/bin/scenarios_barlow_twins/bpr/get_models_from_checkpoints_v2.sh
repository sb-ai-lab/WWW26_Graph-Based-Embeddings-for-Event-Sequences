#!/bin/bash

FREEZE_URL_OPTIONS=("false")
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128)
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.15)
TRAIN_BATCH_SIZE=40

CHECKPOINTS=($(seq 9 10 149))

for FREEZE_URL in "${FREEZE_URL_OPTIONS[@]}"; do
  for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do
    for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
      for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
        for checkpoint_n in "${CHECKPOINTS[@]}"; do

          experiment_name_without_max_epochs="barlow_twins_bpr__batch_size_${TRAIN_BATCH_SIZE}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}"

          ./bin/scenarios_barlow_twins/bpr/get_models_from_checkpoints_single_model.sh \
            "${experiment_name_without_max_epochs}" \
            "${checkpoint_n}" \
        
        done
      done
    done
  done
done