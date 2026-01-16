# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 

BATCH_SIZE=40
SPLIT_COUNT=2
MAX_EPOCHES=150
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128)
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.85 0.15)
FREEZE_URL_OPTIONS=("false")

for freeze_url in "${FREEZE_URL_OPTIONS[@]}"; do
  for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
    for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
      for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do

        experiment_name_without_max_epochs="coles_bpr__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__freeze_url_${freeze_url}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}"
        full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"
        
        echo ${full_experiment_name}

        for checkpoint_n in "${CHECKPOINTS[@]}"; do
          N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
          MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
          MODEL_PATH="models/${MODEL_NAME}.p"

          PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
            --config-dir conf --config-name mles_params_no_url \
            model_path=${MODEL_PATH} \
            logger_name="${experiment_name_without_max_epochs}" \
            \
            +trx_custom_embeddings=ptls_id_to_llm_embedding \
            trx_custom_embeddings.url_host_preprocesed.freeze=${freeze_url} \
            +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
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


          # Extract embeddings
          PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
            --config-dir conf --config-name mles_params_no_url \
            model_path="${MODEL_PATH}" \
            embed_file_name="${MODEL_NAME}_embeddings" \
            inference.batch_size=128 \
            # +inference.devices=0 \
        
        done
      done
    done
  done
done