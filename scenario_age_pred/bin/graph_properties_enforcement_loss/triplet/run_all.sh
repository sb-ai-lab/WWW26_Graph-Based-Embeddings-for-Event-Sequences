TRAIN_BATCH_SIZE=64
SPLIT_COUNT=2
MAX_EPOCHES=150
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128)  # You may also try 1 32 
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.5 0.85 0.15)  # You may also try 0 and 0.01 

for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
  for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
    for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do
      for triplet_loss_margin in "${TRIPLET_LOSS_MARGIN_OPTIONS[@]}"; do


        MODEL_NAME="coles_triplet__batch_size_${TRAIN_BATCH_SIZE}__split_count_${SPLIT_COUNT}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}__triplet_loss_margin_${triplet_loss_margin}__epoches_${MAX_EPOCHES}"
        model_path="models/${MODEL_NAME}_model.p"

        echo "${MODEL_NAME}"

        PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
          --config-dir conf --config-name mles_params \
          data_module=coles_memory_client_id_aware \
          \
          data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
          data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
          ++data_module.train_drop_last=true \
          pl_module.validation_metric.K=1 \
          pl_module.lr_scheduler_partial.step_size=60 \
          model_path="${model_path}" \
          logger_name="${MODEL_NAME}"  \
          \
          trx_embedding_layers=all_features \
          \
          data_module.train_batch_size=${TRAIN_BATCH_SIZE} \
          data_module.train_num_workers=4 \
          data_module.valid_batch_size=64 \
          data_module.valid_num_workers=4  \
          \
          +loss=contrastive_loss_and_additional_loss_convex_combination \
          loss/loss2=triplet \
          loss.loss2.margin=${triplet_loss_margin} \
          loss.alpha=${LOSS_ALPHA} \
          loss.loss2.triplet_selector.num_triplets_per_anchor_user=${N_TRIPLETS_PER_ANCHOR_USER} \
          loss.loss2.triplet_selector.min_elements_in_bin=${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY} \
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
          # device="cpu" \
          # loss/loss2/triplet_selector=bin_triplets_from_similarity_matrix \
          # loss/loss2/triplet_selector/similarity_matrix_slice_getter=actual_matrix \
          

        # Extract embeddings
        PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
          --config-dir conf --config-name mles_params \
          model_path="${model_path}" \
          embed_file_name="${MODEL_NAME}_embeddings" \
          inference.batch_size=128 \
          # +inference.devices=0 \
      
      done
    done
  done
done
