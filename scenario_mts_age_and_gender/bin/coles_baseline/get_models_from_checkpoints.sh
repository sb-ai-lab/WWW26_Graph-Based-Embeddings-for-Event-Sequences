#!/bin/bash

# CHECKPOINTS=(9 19 29 39 49 59 69 79 89 99 109 119 129 139 149)
CHECKPOINTS=($(seq 9 10 149)) 

MAX_EPOCHES=150

BATCH_SIZE=40
SPLIT_COUNT=2

FREEZE_URL_OPTIONS=("true" "false")



for freeze_url in "${FREEZE_URL_OPTIONS[@]}"; do
  experiment_name_without_max_epochs="coles__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__freeze_url_${freeze_url}"
  full_experiment_name="${experiment_name_without_max_epochs}__epoches_${MAX_EPOCHES}"
  
  echo ${full_experiment_name}

  for checkpoint_n in "${CHECKPOINTS[@]}"; do
    N_ECPOCHES_FROM_ONE=$((checkpoint_n+1))
    MODEL_NAME="${experiment_name_without_max_epochs}__epoches_${N_ECPOCHES_FROM_ONE}"
    MODEL_PATH="models/${MODEL_NAME}.p"

    PYTHONPATH=.. python -m ptls_extension_2024_research.utils.torch_model_from_checkpoint \
      --config-dir conf --config-name mles_params_no_url \
      data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
      data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
      pl_module.validation_metric.K=1 \
      pl_module.lr_scheduler_partial.step_size=60 \
      \
      model_path=${MODEL_PATH} \
      logger_name="${experiment_name_without_max_epochs}" \
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
      +trx_custom_embeddings=ptls_id_to_llm_embedding \
      trx_custom_embeddings.url_host_preprocesed.freeze=${freeze_url} \
      +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
      \
      \
      +ckpt_path="./checkpoints/${full_experiment_name}/epoch\=${checkpoint_n}.ckpt" \
      \
      \
      # dataset_unsupervised=parquet_iterable \
      # data_module=ptls_coles_iterable \
      # \
      # device=cpu \
      # hydra.run.dir="hydra_outputs/${experiment_name}" \



    # Execute the Python script for inference
    PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
        model_path=${MODEL_PATH} \
        embed_file_name="${MODEL_NAME}_embeddings" \
        inference.batch_size=64 \
        --config-dir conf --config-name mles_params_no_url
        # +inference.devices=0 \
  
  done
done