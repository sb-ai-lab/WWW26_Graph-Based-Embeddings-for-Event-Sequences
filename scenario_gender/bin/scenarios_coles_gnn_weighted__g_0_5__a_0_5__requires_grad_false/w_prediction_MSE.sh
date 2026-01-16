# Check COLEs with split_count=2
# was 0.637
# python -m ptls.pl_train_module \

LOSS_GAMMA=0.5
GNN_LOSS_ALPHA=0.5

MODEL_NAME="coles_gnn_weighted__w_pred_mse__has_orig__alpha_${GNN_LOSS_ALPHA}__gamma_${LOSS_GAMMA}__requires_grad_false"

model_path="models/${MODEL_NAME}_model.p"


# PYTHONPATH is set to make ptls_extension_2024_research module available
PYTHONPATH=.. python -m ptls.pl_train_module \
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
    pl_module.loss_gamma=${LOSS_GAMMA} \
    pl_module.gnn_loss_alpha=${GNN_LOSS_ALPHA} \
    gnn_link_predictor.use_edge_weights="true" \
    gnn_link_predictor.link_predictor_name="one_layer" \
    pl_module.lp_criterion_name="MSELoss" \
    gnn_link_predictor.link_predictor_add_sigmoid="false" \
    pl_module.freeze_embeddings_outside_coles_batch="true" \
    trainer.max_epochs=40 \
    # device="cpu"


PYTHONPATH=.. python -m pl_inference_with_client_id    \
    model_path="${model_path}" \
    embed_file_name="${MODEL_NAME}_embeddings" \
    inference.batch_size=32 \
    --config-dir conf --config-name coles_gnn_end2end_params_full_graph 
