# Define the options lists

MAX_EPOCHES=40
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(32 128)
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0 0.5 0.85 0.15 0.01)

for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
    for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
        for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do

            # Construct the MODEL_NAME and model_path
            MODEL_NAME="coles_bpr__loss_alpha_${LOSS_ALPHA}__${MAX_EPOCHES}__epoches__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}__try_2"
            model_path="models/${MODEL_NAME}_model.p"

            # Train pretext task
            PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
                --config-dir conf --config-name coles_bpr_params \
                data_module.train_data.splitter.split_count=2 \
                data_module.valid_data.splitter.split_count=2 \
                pl_module.validation_metric.K=1 \
                pl_module.lr_scheduler_partial.step_size=60 \
                model_path="${model_path}" \
                logger_name="${MODEL_NAME}"  \
                data_module.train_batch_size=64 \
                data_module.train_num_workers=4 \
                data_module.valid_batch_size=64 \
                data_module.valid_num_workers=4  \
                \
                pl_module.loss.alpha=${LOSS_ALPHA} \
                pl_module.loss.loss2.triplet_selector.num_triplets_per_anchor_user=${N_TRIPLETS_PER_ANCHOR_USER} \
                pl_module.loss.loss2.triplet_selector.bin_separation_strategy.min_elements_in_bin=${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY} \
                \
                +trainer.checkpoints_every_n_val_epochs=10 \
                +trainer.checkpoint_dirpath="./checkpoints/${MODEL_NAME}" \
                +trainer.checkpoint_filename="\{epoch\}" \
                trainer.max_epochs="${MAX_EPOCHES}" \
                \
                hydra.run.dir="hydra_outputs/${MODEL_NAME}" \
                # device="cpu"
                


            # Extract embeddings
            PYTHONPATH=.. python -m pl_inference_with_client_id \
                model_path="${model_path}" \
                embed_file_name="${MODEL_NAME}_embeddings" \
                inference.batch_size=32 \
                --config-dir conf --config-name coles_bpr_params 


        done
    done
done
