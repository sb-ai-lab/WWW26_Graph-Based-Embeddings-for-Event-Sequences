#!/bin/bash

FREEZE_URL_OPTIONS=("false")
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128)
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.15)

for FREEZE_URL in "${FREEZE_URL_OPTIONS[@]}"; do
  for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
    for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
      for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do
        ./bin/scenarios_barlow_twins/bpr/run_single.sh \
            "${LOSS_ALPHA}" \
            "${N_TRIPLETS_PER_ANCHOR_USER}" \
            "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}" \
            "${FREEZE_URL}" \

      done
    done
  done
done