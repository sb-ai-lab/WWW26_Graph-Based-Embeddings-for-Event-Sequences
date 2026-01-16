#!/bin/bash

MAX_EPOCHES=150
BATCH_SIZE=40
FREEZE_URL="false"


experiment_name="barlow__batch_size_${BATCH_SIZE}__freeze_url_${FREEZE_URL}__epoches_${MAX_EPOCHES}"


echo ""
echo "EXPERIMENT: ${experiment_name}"
echo ""


PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
    --config-dir conf --config-name barlow_twins_params_no_url \
    \
    data_module.train_data.splitter.split_count=2 \
    data_module.valid_data.splitter.split_count=2 \
    \
    model_path="models/${experiment_name}.p" \
    logger_name="${experiment_name}" \
    \
    data_module.train_batch_size=${BATCH_SIZE} \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=40 \
    data_module.valid_num_workers=4 \
    \
    +trainer.checkpoints_every_n_val_epochs=10 \
    +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
    +trainer.checkpoint_filename="\{epoch\}" \
    trainer.max_epochs=${MAX_EPOCHES} \
    \
    hydra.run.dir="hydra_outputs/${experiment_name}" \
    \
    +trx_custom_embeddings=ptls_id_to_llm_embedding \
    trx_custom_embeddings.url_host_preprocesed.freeze=${FREEZE_URL} \
    +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
    \
    # dataset_unsupervised=parquet_iterable \
    # data_module=ptls_coles_iterable \
    # \
    # device=cpu



PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
    model_path="models/${experiment_name}.p" \
    embed_file_name="${experiment_name}_embeddings" \
    inference.batch_size=40 \
    --config-dir conf --config-name barlow_twins_params_no_url
