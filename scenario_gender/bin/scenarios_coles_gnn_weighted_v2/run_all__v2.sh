#! This is not a script that was originally launched. It's an attempt to gather all optional in one script.


# INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS=(true false)
INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS=(false)
# HAS_ORIG_EMBS_OPTIONS=(true false)
HAS_ORIG_EMBS_OPTIONS=(false)
# GNN_NAME_OPTIONS=("GraphSAGE" "GAT")
GNN_NAME_OPTIONS=("GraphSAGE")
# COLES_LOSS_GAMMA_OPTIONS=(0.5 0.8)
COLES_LOSS_GAMMA_OPTIONS=(0.5)
# GNN_LOSS_ALPHA_OPTIONS=(0.5 0.25)
GNN_LOSS_ALPHA_OPTIONS=(0.5 0.25)
# LP_CRITERION_OPTIONS=("MSELoss" "BCELoss")
LP_CRITERION_OPTIONS=("MSELoss")


MAX_EPOCHES=40

for INCLUDE_GNN_USERS_IN_COLES_LOSS in "${INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS[@]}"; do
    for HAS_ORIG_EMBS in "${HAS_ORIG_EMBS_OPTIONS[@]}"; do
        for GNN_NAME in "${GNN_NAME_OPTIONS[@]}"; do
            for COLES_LOSS_GAMMA in "${COLES_LOSS_GAMMA_OPTIONS[@]}"; do
                for GNN_LOSS_ALPHA in "${GNN_LOSS_ALPHA_OPTIONS[@]}"; do
                        for LP_CRITERION in "${LP_CRITERION_OPTIONS[@]}"; do


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


                        MODEL_NAME="coles_gnn_weighted__w_pred___${GNN_NAME}__${HAS_ORIG_EMBS_STR}__alpha_${GNN_LOSS_ALPHA}__gamma_${COLES_LOSS_GAMMA}__${INCLUDE_GNN_USERS_IN_COLES_LOSS_STR}__lp_criterion_${LP_CRITERION}__${MAX_EPOCHES}_epoches"


                        model_path="models/${MODEL_NAME}_model.p"

                        # PYTHONPATH is set to make ptls_extension_2024_research module available
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
                            pl_module.loss_gamma="${COLES_LOSS_GAMMA}" \
                            pl_module.gnn_loss_alpha="${GNN_LOSS_ALPHA}" \
                            pl_module.use_gnn_loss="true" \
                            gnn_link_predictor.use_edge_weights="true" \
                            gnn_link_predictor.link_predictor_name="one_layer" \
                            pl_module.lp_criterion_name="${LP_CRITERION}" \
                            gnn_link_predictor.link_predictor_add_sigmoid="false" \
                            gnn_link_predictor.gnn_name="${GNN_NAME}" \
                            ckpt_callback.every_n_epochs=5 \
                            ckpt_callback.dirpath="./checkpoints/${MODEL_NAME}" \
                            +trainer.additional_callbacks=[\${ckpt_callback}] \
                            +trx_simple_embedding_layers="${TRX_SIMPLE_EMBEDDING_LAYERS}" \
                            trainer.max_epochs="${MAX_EPOCHES}" \
                            device="cpu"


                        # PYTHONPATH=.. python -m pl_inference_with_client_id    \
                        #     model_path="${model_path}" \
                        #     embed_file_name="${MODEL_NAME}_embeddings" \
                        #     inference.batch_size=32 \
                        #     +inference.devices=0 \
                        #     --config-dir conf --config-name coles_gnn_end2end_params_full_graph \

                            # +trainer.enable_checkpointing="true" \

                    done
                done
            done
        done
    done
done



for INCLUDE_GNN_USERS_IN_COLES_LOSS in "${INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS[@]}"; do
    for HAS_ORIG_EMBS in "${HAS_ORIG_EMBS_OPTIONS[@]}"; do
        for GNN_NAME in "${GNN_NAME_OPTIONS[@]}"; do
            for COLES_LOSS_GAMMA in "${COLES_LOSS_GAMMA_OPTIONS[@]}"; do
                for GNN_LOSS_ALPHA in "${GNN_LOSS_ALPHA_OPTIONS[@]}"; do
                        for LP_CRITERION in "${LP_CRITERION_OPTIONS[@]}"; do

                        PYTHONPATH=.. python -m pl_inference_with_client_id    \
                            model_path="${model_path}" \
                            embed_file_name="${MODEL_NAME}_embeddings" \
                            inference.batch_size=32 \
                            +inference.devices=0 \
                            --config-dir conf --config-name coles_gnn_end2end_params_full_graph \
                    
                    done
                done
            done
        done
    done
done