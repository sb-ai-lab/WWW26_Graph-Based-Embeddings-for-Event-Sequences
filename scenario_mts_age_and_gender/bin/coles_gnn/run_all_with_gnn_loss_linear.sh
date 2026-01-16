

# INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS=(true false)
INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS=(false)
# GNN_NAME_OPTIONS=("GraphSAGE" "GAT")
GNN_NAME_OPTIONS=("GraphSAGE")
# COLES_LOSS_GAMMA_OPTIONS=(0.5 0.8)
COLES_LOSS_GAMMA_OPTIONS=(0.5)
# GNN_LOSS_ALPHA_OPTIONS=(0.5 0.25)
GNN_LOSS_ALPHA_OPTIONS=(0.5)
# LP_CRITERION_OPTIONS=("MSELoss" "BCELoss")
LP_CRITERION_OPTIONS=("MSELoss")


TRX_SIMPLE_EMBEDDING_LAYERS="all_except_item_id"

MAX_EPOCHES=150

BATCH_SIZE=256
SPLIT_COUNT=2
GNN_OUTPUT_EMBEDDING_SIZE=48

# !!!! GNN_INPUT_EMBEDDING_SIZE has to be altered ONLY if gnn_embedder=item_embedding_has_linear is provided
# otherwise it MUST be equal  to llm_embedding_size (256)
GNN_INPUT_EMBEDDING_SIZE=48





nvidia-smi




for INCLUDE_GNN_USERS_IN_COLES_LOSS in "${INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS[@]}"; do
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


          MODEL_NAME="coles_gnn__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__gnn_${GNN_NAME}__alpha_${GNN_LOSS_ALPHA}__gamma_${COLES_LOSS_GAMMA}__${INCLUDE_GNN_USERS_IN_COLES_LOSS_STR}__lp_criterion_${LP_CRITERION}__gnn_out_size_${GNN_OUTPUT_EMBEDDING_SIZE}__epoches_${MAX_EPOCHES}"


          model_path="models/${MODEL_NAME}_model.p"

          # PYTHONPATH is set to make ptls_extension_2024_research module available
          PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
            --config-dir conf --config-name coles_gnn_params \
            data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
            data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
            pl_module.coles_validation_metric.K=1 \
            pl_module.lr_scheduler_partial.step_size=60 \
            \
            model_path="${model_path}" \
            logger_name="${MODEL_NAME}"  \
            \
            data_module.train_batch_size=${BATCH_SIZE} \
            data_module.train_num_workers=4 \
            data_module.valid_batch_size=64 \
            data_module.valid_num_workers=4  \
            \
            pl_module.include_gnn_users_in_contrastive_loss="${INCLUDE_GNN_USERS_IN_COLES_LOSS}" \
            pl_module.loss_gamma="${COLES_LOSS_GAMMA}" \
            pl_module.gnn_loss_alpha="${GNN_LOSS_ALPHA}" \
            pl_module.use_gnn_loss="true" \
            \
            gnn_embedder.use_edge_weights="true" \
            \
            pl_module.link_predictor_name="one_layer" \
            pl_module.lp_criterion_name="${LP_CRITERION}" \
            pl_module.link_predictor_add_sigmoid="false" \
            gnn_embedder.gnn_name="${GNN_NAME}" \
            \
            +trainer.checkpoints_every_n_val_epochs=10 \
            +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
            +trainer.checkpoint_filename="\{epoch\}" \
            \
            hydra.run.dir="hydra_outputs/${MODEL_NAME}" \
            \
            trx_embedding_layers="${TRX_SIMPLE_EMBEDDING_LAYERS}" \
            \
            gnn_output_embedding_size=${GNN_OUTPUT_EMBEDDING_SIZE} \
            \
            trainer.max_epochs="${MAX_EPOCHES}" \
            \
            +additional_artifacts_to_save="[git_commit_hash, full_pl_module]" \
            \
            gnn_input_embedding_size="${GNN_INPUT_EMBEDDING_SIZE}" \
            gnn_embedder=item_embedding_has_linear \
            # \
            # dataset_unsupervised=parquet_debug \
            # graph_path=data/graphs/weighted_subgraph_for_test_with_15000_users_and_all_items \
            # num_users=15000 \
            # \
            # device="cpu" \
            # \
            # dataset_unsupervised=parquet_iterable \
            # data_module=ptls_coles_iterable \
    



          # PYTHONPATH=.. python -m pl_inference_with_client_id  \
          #   model_path="${model_path}" \
          #   embed_file_name="${MODEL_NAME}_embeddings" \
          #   inference.batch_size=32 \
          #   +inference.devices=0 \
          #   --config-dir conf --config-name coles_gnn_end2end_params_full_graph \

            # +trainer.enable_checkpointing="true" \

        done
      done
    done
  done
done



for INCLUDE_GNN_USERS_IN_COLES_LOSS in "${INCLUDE_GNN_USERS_IN_COLES_LOSS_OPTIONS[@]}"; do
    for GNN_NAME in "${GNN_NAME_OPTIONS[@]}"; do
      for COLES_LOSS_GAMMA in "${COLES_LOSS_GAMMA_OPTIONS[@]}"; do
        for GNN_LOSS_ALPHA in "${GNN_LOSS_ALPHA_OPTIONS[@]}"; do
            for LP_CRITERION in "${LP_CRITERION_OPTIONS[@]}"; do

              PYTHONPATH=.. python -m pl_inference_with_client_id  \
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