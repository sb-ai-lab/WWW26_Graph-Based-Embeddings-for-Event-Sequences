#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 139)) 


BATCH_SIZE=64
SPLIT_COUNT=2
MAX_EPOCHES=150
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128)
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.85 0.5 0.15)
TRIPLET_LOSS_MARGIN_OPTIONS=(1.0 0.5 2.0 4.0 6.0)


for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
  for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
    for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do
      for triplet_loss_margin in "${TRIPLET_LOSS_MARGIN_OPTIONS[@]}"; do


          experiment_name_without_max_epochs="coles_triplet__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}__triplet_loss_margin_${triplet_loss_margin}"
          full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"
        

          echo "${full_experiment_name}"

          for checkpoint_n in "${CHECKPOINTS[@]}"; do
          N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
          MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
          MODEL_PATH="models/${MODEL_NAME}_model.p"
          
          PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
            --config-dir conf --config-name mles_params \
            pl_module.validation_metric.K=1 \
            pl_module.lr_scheduler_partial.step_size=60 \
            model_path="${MODEL_PATH}" \
            logger_name="${MODEL_NAME}"  \
            \
            trx_embedding_layers=all_features \
            \
            +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
            \
            # device="cpu"
            

          # Extract embeddings
          PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
            --config-dir conf --config-name mles_params \
            model_path="${MODEL_PATH}" \
            embed_file_name="${MODEL_NAME}_embeddings" \
            inference.batch_size=256 \
            # +inference.devices=0 \

        done
      done
    done
  done
done
