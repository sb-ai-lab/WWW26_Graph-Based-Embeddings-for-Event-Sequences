#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 
MAX_EPOCHES=150
BATCH_SIZE=40

FREEZE_URL_OPTIONS=("false")


for freeze_url in "${FREEZE_URL_OPTIONS[@]}"; do
  experiment_name_without_max_epochs="barlow__batch_size_${BATCH_SIZE}__freeze_url_${freeze_url}"
  full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"
  
  echo ${full_experiment_name}

  for checkpoint_n in "${CHECKPOINTS[@]}"; do
    ./get_models_from_checkpoints_single_model.sh \
      "${freeze_url}" \
      "${experiment_name_without_max_epochs}" \
      "${full_experiment_name}" \
      "${checkpoint_n}"
  
  done
done