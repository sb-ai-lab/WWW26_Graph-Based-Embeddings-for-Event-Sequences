#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 139)) 

TRAIN_BATCH_SIZE=64
SPLIT_COUNT=2
MAX_EPOCHES=150
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128)  # You may also try 1 and 32 
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.5 0.85 0.15)  # You may also try 0 and 0.01 


for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
  for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
    for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do

      experiment_name_without_max_epochs="coles_bpr__batch_size_${TRAIN_BATCH_SIZE}__split_count_${SPLIT_COUNT}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}"
      full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}__try_2"
      
      echo ${full_experiment_name}

      for checkpoint_n in "${CHECKPOINTS[@]}"; do
        N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
        MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}__try_2"
        MODEL_PATH="models/${MODEL_NAME}.p"

        PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
          --config-dir conf --config-name mles_params_client_id_aware \
          data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
          data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
          pl_module.validation_metric.K=1 \
          pl_module.lr_scheduler_partial.step_size=60 \
          \
          model_path=${MODEL_PATH} \
          logger_name="${experiment_name_without_max_epochs}" \
          \
          data_module.train_batch_size=${TRAIN_BATCH_SIZE} \
          data_module.train_num_workers=4 \
          data_module.valid_batch_size=64 \
          data_module.valid_num_workers=4  \
          \
          +loss=contrastive_loss_and_additional_loss_convex_combination \
          loss.alpha=${LOSS_ALPHA} \
          loss.loss2.triplet_selector.num_triplets_per_anchor_user=${N_TRIPLETS_PER_ANCHOR_USER} \
          loss.loss2.triplet_selector.min_elements_in_bin=${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY} \
          pl_module.loss=\${loss} \
          \
          +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
          \
          # device="cpu" \


        # Execute the Python script for inference
        PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference_with_client_id \
            model_path=${MODEL_PATH} \
            embed_file_name="${MODEL_NAME}_embeddings" \
            inference.batch_size=64 \
            +inference.devices=0 \
            --config-dir conf --config-name mles_params_client_id_aware
            # +inference.devices=0 \

      done
    done
  done
done