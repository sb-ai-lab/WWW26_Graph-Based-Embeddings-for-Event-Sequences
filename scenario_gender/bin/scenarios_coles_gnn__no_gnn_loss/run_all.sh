# Define the options lists
INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS=(true false)
HAS_ORIG_EMBS_OPTIONS=(true false)
GNN_NAME_OPTIONS=("GraphSAGE" "GAT")

MAX_EPOCHES=40

for INCLUDE_GNN_USERS_IN_COLES_LOSS in "${INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS[@]}"; do
    for HAS_ORIG_EMBS in "${HAS_ORIG_EMBS_OPTIONS[@]}"; do
        for GNN_NAME in "${GNN_NAME_OPTIONS[@]}"; do

            # Determine INCLUDE_GNN_USERS_IN_COLES_LOSS_STR based on the value of INCLUDE_GNN_USERS_IN_COLES_LOSS
            if [ "$INCLUDE_GNN_USERS_IN_COLES_LOSS" = true ]; then
                INCLUDE_GNN_USERS_IN_COLES_LOSS_STR="gnn_users_in_coles_loss"
            else
                INCLUDE_GNN_USERS_IN_COLES_LOSS_STR=""
            fi

            # Determine HAS_ORIG_EMBS_STR and TRX_SIMPLE_EMBEDDING_LAYERS based on the value of HAS_ORIG_EMBS
            if [ "$HAS_ORIG_EMBS" = true ]; then
                HAS_ORIG_EMBS_STR="has_orig"
                TRX_SIMPLE_EMBEDDING_LAYERS="all_features"
            else
                HAS_ORIG_EMBS_STR="no_orig"
                TRX_SIMPLE_EMBEDDING_LAYERS="all_except_item_id"
            fi

            # Construct the MODEL_NAME and model_path
            MODEL_NAME="coles_gnn_weighted_2__${GNN_NAME}__no_coles_loss__${HAS_ORIG_EMBS_STR}__${INCLUDE_GNN_USERS_IN_COLES_LOSS_STR}__${MAX_EPOCHES}_epoches"
            model_path="models/${MODEL_NAME}_model.p"

            # Train pretext task
            PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
                --config-dir conf --config-name coles_gnn_end2end_params_full_graph \
                data_module.train_data.splitter.split_count=2 \
                data_module.valid_data.splitter.split_count=2 \
                pl_module.coles_validation_metric.K=1 \
                pl_module.lr_scheduler_partial.step_size=60 \
                model_path="${model_path}" \
                logger_name="${MODEL_NAME}"  \
                data_module.train_batch_size=64 \
                data_module.train_num_workers=4 \
                data_module.valid_batch_size=64 \
                data_module.valid_num_workers=4  \
                pl_module.include_gnn_users_in_contrastive_loss="${INCLUDE_GNN_USERS_IN_COLES_LOSS}" \
                pl_module.use_gnn_loss="false" \
                gnn_link_predictor.gnn_name="${GNN_NAME}" \
                +trainer.checkpoints_every_n_val_epochs=10 \
                +trainer.checkpoint_dirpath="./checkpoints/${MODEL_NAME}" \
                +trainer.checkpoint_filename="\{epoch\}" \
                +trx_simple_embedding_layers="${TRX_SIMPLE_EMBEDDING_LAYERS}" \
                trainer.max_epochs="${MAX_EPOCHES}" \
                # device="cpu"


            # Extract embeddings
            PYTHONPATH=.. python -m pl_inference_with_client_id \
                model_path="${model_path}" \
                embed_file_name="${MODEL_NAME}_embeddings" \
                inference.batch_size=32 \
                --config-dir conf --config-name coles_gnn_end2end_params_full_graph 

        done
    done
done
