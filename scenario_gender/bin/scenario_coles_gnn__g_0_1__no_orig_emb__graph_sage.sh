# Check COLEs with split_count=2
# was 0.637
# python -m ptls.pl_train_module \

# PYTHONPATH is set to make ptls_extension_2024_research module available
PYTHONPATH=.. python -m ptls.pl_train_module \
    --config-dir conf --config-name coles_gnn_end2end_params_full_graph_no_orig_emb  \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    pl_module.coles_validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="models/coles_gnn__g_0_1__no_orig_emb__sample_2__model.p" \
    logger_name="coles_gnn__g_0_1__no_orig_emb_2"  \
    data_module.train_batch_size=64 \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=64 \
    data_module.valid_num_workers=4  \
    pl_module.loss_gamma=0.1 \
    trainer.max_epochs=40
    # trainer.max_epochs=2
    

PYTHONPATH=.. python -m pl_inference_with_client_id    \
    model_path="models/coles_gnn__g_0_1__no_orig_emb__sample_2__model.p" \
    embed_file_name="coles_gnn__g_0_1__no_orig_emb__sample_2__embeddings" \
    inference.batch_size=32 \
    --config-dir conf --config-name coles_gnn_end2end_params_full_graph_no_orig_emb  

    # +inference.devices=0 \
