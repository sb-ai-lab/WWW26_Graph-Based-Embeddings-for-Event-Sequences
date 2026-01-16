#!/bin/bash

# COLEs with split_count=2
MAX_EPOCHES=150

BATCH_SIZE=40
SPLIT_COUNT=2


FREEZE_URL_OPTIONS=("true" "false")


for freeze_url in "${FREEZE_URL_OPTIONS[@]}"; do


    experiment_name="coles__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__freeze_url_${freeze_url}__epoches_${MAX_EPOCHES}"


    echo ""
    echo "EXPERIMENT: ${experiment_name}"
    echo ""


    PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
        --config-dir conf --config-name mles_params_no_url \
        data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
        data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
        pl_module.validation_metric.K=1 \
        pl_module.lr_scheduler_partial.step_size=60 \
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
        trx_custom_embeddings.url_host_preprocesed.freeze=${freeze_url} \
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
        --config-dir conf --config-name mles_params_no_url

done