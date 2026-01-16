#!/bin/bash

MAX_EPOCHES=150
BATCH_SIZE=64
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128 32) 
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.5 0.85 0.15 0.01 0) 

CHECKPOINTS=($(seq 9 10 149))

for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do
  for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
    for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
      ./get_models_from_checkpoints_single_model.sh \
        "${LOSS_ALPHA}" \
        "${N_TRIPLETS_PER_ANCHOR_USER}" \
        "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}" \
        "${CHECKPOINTS[*]}"  # Pass checkpoints as a space-separated string
      
    done
  done
done