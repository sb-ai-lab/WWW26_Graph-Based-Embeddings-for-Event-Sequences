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
GNN_LOSS_ALPHA_OPTIONS=(0.5)
# LP_CRITERION_OPTIONS=("MSELoss" "BCELoss")
LP_CRITERION_OPTIONS=("MSELoss")

MAX_EPOCHES_ORIG=40


# CKPT_EPOCHS=(19 24) 
CKPT_EPOCHS=(29 4 9 14 34)


for INCLUDE_GNN_USERS_IN_COLES_LOSS in "${INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS[@]}"; do
    for HAS_ORIG_EMBS in "${HAS_ORIG_EMBS_OPTIONS[@]}"; do
        for GNN_NAME in "${GNN_NAME_OPTIONS[@]}"; do
            for COLES_LOSS_GAMMA in "${COLES_LOSS_GAMMA_OPTIONS[@]}"; do
                for GNN_LOSS_ALPHA in "${GNN_LOSS_ALPHA_OPTIONS[@]}"; do
                        for LP_CRITERION in "${LP_CRITERION_OPTIONS[@]}"; do
                            for CKPT_EPOCH in "${CKPT_EPOCHS[@]}"; do

                                MAX_EPOCHES=$((MAX_EPOCHES_ORIG - CKPT_EPOCH - 1))
                                echo ${MAX_EPOCHES}


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

                                ORIGINAL_MODEL_NAME="coles_gnn_weighted__w_pred___${GNN_NAME}__${HAS_ORIG_EMBS_STR}__alpha_${GNN_LOSS_ALPHA}__gamma_${COLES_LOSS_GAMMA}__${INCLUDE_GNN_USERS_IN_COLES_LOSS_STR}__lp_criterion_${LP_CRITERION}__${MAX_EPOCHES_ORIG}_epoches"

                                MODEL_NAME="check_coles_only__pretrained_epoches_${CKPT_EPOCH}__coles_gnn_weighted__w_pred__no_gnn_${GNN_NAME}__${HAS_ORIG_EMBS_STR}__alpha_${GNN_LOSS_ALPHA}__gamma_${COLES_LOSS_GAMMA}__${INCLUDE_GNN_USERS_IN_COLES_LOSS_STR}__lp_criterion_${LP_CRITERION}__${MAX_EPOCHES_ORIG}_epoches__epoches_after_ckpt_${MAX_EPOCHES}__try_1"


                                model_path="models/${MODEL_NAME}_model.p"



                                PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
                                --config-dir conf --config-name coles_gnn_pretrained_params__client_id_aware \
                                \
                                data_module.train_data.splitter.split_count=2 \
                                data_module.valid_data.splitter.split_count=2 \
                                pl_module.validation_metric.K=1 \
                                pl_module.lr_scheduler_partial.step_size=60 \
                                \
                                +pl_module.seq_encoder.trx_encoder.use_batch_norm="false" \
                                \
                                model_path="${model_path}" \
                                logger_name="${MODEL_NAME}"  \
                                data_module.train_batch_size=64 \
                                data_module.train_num_workers=4 \
                                data_module.valid_batch_size=64 \
                                data_module.valid_num_workers=4  \
                                +pretrained_embs_init_state="{_target_: torch.zeros, size: {_target_: ptls_extension_2024_research.hydra_utils.to_list,  x: [14344, 48]}}" \
                                pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings="\${pretrained_embs_init_state}" \
                                \
                                +model_weights_only_ckpt="./checkpoints/converted_to_pretrained/${ORIGINAL_MODEL_NAME}/pretrain_epoches_${CKPT_EPOCH}.ckpt" \
                                \
                                trainer.max_epochs="${MAX_EPOCHES}" \
                                \
                                # ckpt_callback.every_n_epochs=1 \
                                # ckpt_callback.dirpath="./checkpoints/${MODEL_NAME}" \
                                # +trainer.additional_callbacks=[\${ckpt_callback}] \
                                
                                # device="cpu"

                                # pl_module.include_gnn_users_in_contrastive_loss="${INCLUDE_GNN_USERS_IN_COLES_LOSS}" \
                                # +trainer.resume_checkpoint_path="./checkpoints/converted_to_pretrained/${ORIGINAL_MODEL_NAME}/pretrain_epoches_${CKPT_EPOCH}.ckpt" \




                                PYTHONPATH=.. python -m pl_inference_with_client_id    \
                                model_path="${model_path}" \
                                embed_file_name="${MODEL_NAME}_embeddings" \
                                inference.batch_size=24 \
                                --config-dir conf --config-name coles_gnn_end2end_params_full_graph \


                        done
                    done
                done
            done
        done
    done
done