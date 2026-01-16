#!/bin/bash

# Set the root directory for pretrained models
PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

# Set the maximum number of epochs
MAX_EPOCHES=150

# Declare an associative array to map model directories to epoch lists
declare -A model_epoch_map

# Populate the associative array with model directories and corresponding epoch lists
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0"]="100 " 

# Iterate over each model directory
for model_dir in "${!model_epoch_map[@]}"; do
  # Get the list of epochs for the current model directory
  IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

  # Iterate over each epoch number
  for epoch in "${EPOCHES[@]}"; do

    # Construct the file name and paths
    f_name="${epoch}.pt"
    embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

    # Define the experiment name
    experiment_name="coles_gnn__pretrained_${model_dir}__pretrain_epoches_${epoch}__epoches_${MAX_EPOCHES}"

    echo ""
    echo "EXPERIMENT: ${experiment_name}"
    echo ""

    PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
      --config-dir conf --config-name coles_gnn_pretrained_params \
      data_module.train_data.splitter.split_count=2 \
      data_module.valid_data.splitter.split_count=2 \
      pl_module.validation_metric.K=1 \
      pl_module.lr_scheduler_partial.step_size=60 \
      model_path="models/${experiment_name}.p" \
      logger_name="${experiment_name}" \
      data_module.train_batch_size=64 \
      data_module.train_num_workers=4 \
      data_module.valid_batch_size=64 \
      data_module.valid_num_workers=4 \
      pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings_path="${embeddings_path}" \
      +trainer.checkpoints_every_n_val_epochs=10 \
      +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
      +trainer.checkpoint_filename="\{epoch\}" \
      trainer.max_epochs=${MAX_EPOCHES} \
    #   device="cpu"

        # +trainer.checkpoint_filename='{epoch}-{val_loss:.3f}-{val_word_level_accuracy:.3f}')


    # Execute the Python script for inference
    PYTHONPATH=.. python -m ptls.pl_inference \
      model_path="models/${experiment_name}.p" \
      embed_file_name="${experiment_name}_embeddings" \
      inference.batch_size=16 \
      --config-dir conf --config-name coles_gnn_pretrained_params

      # +inference.devices=0 \

  done
done


# # Compare
# rm results/scenario_gender_2024_research__pretrained_gnn.txt
# # rm -r conf/embeddings_validation.work/
# python -m embeddings_validation \
#     --config-dir conf --config-name embeddings_validation__pretrained_gnn +workers=10 +total_cpu_count=4
