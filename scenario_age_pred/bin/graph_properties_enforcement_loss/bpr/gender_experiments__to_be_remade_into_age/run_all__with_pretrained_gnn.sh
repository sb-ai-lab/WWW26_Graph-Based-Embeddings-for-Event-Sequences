PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'



declare -A model_epoch_map
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0"]="100 75" # "75 100 150 200" 
model_epoch_map["wl-0.5_gnn-GAT_res-True_wd-0"]="75"  # "15 50 100 150 200" 
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.01"]="50"  #"25 50 75 100 150 200"  
model_epoch_map["wl-0.5_gnn-GAT_res-True_wd-0.01"]="100" #"25 50 100 150 200"
# model_epoch_map["wl-0.5_gnn-GAT_res-True_wd-0.1"]="25 50 75 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GraphSAGE_res-False_wd-0"]="50 75 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.1"]="50 75 100 150 200" 



MAX_EPOCHES=40
N_TRIPLETS_PER_ANCHOR_USER_OPTIONS=(128 32 8 1)
MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS=(5)
LOSS_ALPHA_OPTIONS=(0.5 0.85 0.15)

for model_dir in "${!model_epoch_map[@]}"; do
    # Get the list of epochs for the current model directory
    IFS=' ' read -r -a PRETRAIN_EPOCHs_LST <<< "${model_epoch_map[$model_dir]}"

    for pretrain_epoch in "${PRETRAIN_EPOCHs_LST[@]}"; do 

        embeddings_f_name="${pretrain_epoch}.pt"
        embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${embeddings_f_name}"
        
        for MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY in "${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY_OPTIONS[@]}"; do
            for N_TRIPLETS_PER_ANCHOR_USER in "${N_TRIPLETS_PER_ANCHOR_USER_OPTIONS[@]}"; do
                for LOSS_ALPHA in "${LOSS_ALPHA_OPTIONS[@]}"; do



                    # Construct the experiment_name and model_path
                    experiment_name="check__coles_bpr_with_pretrained_gnn__GNN_${model_dir}__pretrain_epoches_${pretrain_epoch}__loss_alpha_${LOSS_ALPHA}__triplets_per_user_${N_TRIPLETS_PER_ANCHOR_USER}__bin_separation_margin_${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY}__${MAX_EPOCHES}__epoches__try_1"
                    model_path="models/${experiment_name}_model.p"

                    echo ""
                    echo "EXPERIMENT: ${experiment_name}"
                    echo ""

                    # Train pretext task
                    PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
                        --config-dir conf --config-name coles_bpr_with_pretrained_gnn_params \
                        data_module.train_data.splitter.split_count=2 \
                        data_module.valid_data.splitter.split_count=2 \
                        pl_module.validation_metric.K=1 \
                        pl_module.lr_scheduler_partial.step_size=60 \
                        model_path="${model_path}" \
                        logger_name="${experiment_name}"  \
                        data_module.train_batch_size=64 \
                        data_module.train_num_workers=4 \
                        data_module.valid_batch_size=64 \
                        data_module.valid_num_workers=4  \
                        \
                        pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
                        \
                        pl_module.loss.alpha=${LOSS_ALPHA} \
                        pl_module.loss.loss2.triplet_selector.num_triplets_per_anchor_user=${N_TRIPLETS_PER_ANCHOR_USER} \
                        pl_module.loss.loss2.triplet_selector.bin_separation_strategy.min_elements_in_bin=${MIN_ELEMENTS_IN_BIN_WHEN_ONE_BIN_ONLY} \
                        \
                        +trainer.checkpoints_every_n_val_epochs=10 \
                        +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
                        +trainer.checkpoint_filename="\{epoch\}" \
                        trainer.max_epochs="${MAX_EPOCHES}" \
                        # device="cpu"


                        # Extract embeddings
                        PYTHONPATH=.. python -m pl_inference_with_client_id \
                            model_path="${model_path}" \
                            embed_file_name="${experiment_name}_embeddings" \
                            inference.batch_size=32 \
                            --config-dir conf --config-name coles_bpr_with_pretrained_gnn_params 

                done
            done
        done
    done
done
