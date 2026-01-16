#!/bin/bash

# Set the root directory for pretrained models
PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

MAX_EPOCHES=40

declare -A model_epoch_map

# Populate the associative array with model directories and corresponding epoch lists
# model_epoch_map["wl-0.5_gnn-GAT_res-True_wd-0"]="15 50 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GAT_res-True_wd-0.1"]="25 50 75 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GAT_res-True_wd-0.01"]="25 50 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GraphSAGE_res-False_wd-0"]="50 75 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0"]="75 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.1"]="50 75 100 150 200" 
# model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.01"]="25 50 75 100 150 200" 
# model_epoch_map["GRACE-GAT-weight-decay_0.00001__residual_true__num_layers_3__emb_size_128"]="11" 
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_32"]="0 2 4 6 8 10 20 50 100 150 200"
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_128"]="0 2 4 6 8 10 20 50 100 150 200"
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_64"]="0 2 4 6 8 10 20 50 100 150 200"


for model_dir in "${!model_epoch_map[@]}"; do
  IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

  for epoch in "${EPOCHES[@]}"; do

    f_name="${epoch}.pt"
    embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

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
      pl_module.seq_encoder.trx_encoder.custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
      +trainer.checkpoints_every_n_val_epochs=10 \
      +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
      +trainer.checkpoint_filename="\{epoch\}" \
      trainer.max_epochs=${MAX_EPOCHES} \
      \
      hydra.run.dir="hydra_outputs/${experiment_name}" \
      \
      +additional_artifacts_to_save="[git_commit_hash]" \
      \
      # device="cpu"



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
