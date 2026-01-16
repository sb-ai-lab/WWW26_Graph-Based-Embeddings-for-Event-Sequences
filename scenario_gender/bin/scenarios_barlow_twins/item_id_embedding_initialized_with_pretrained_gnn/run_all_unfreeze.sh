#!/bin/bash

PRETRAINED_MODELS_ROOT='models/pretrained_gnn_models'

MAX_EPOCHES=150

BATCH_SIZE=64
FREEZE_PRETRAINED_EMBS_OPTIONS=(false)



declare -A model_epoch_map
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_64"]="15 51 101 151 191"
model_epoch_map["wl-0.5_gnn-GraphSAGE_res-True_wd-0.0__emb_size_128"]="15 51 101 151 191"


for freeze_pretrained_embs in "${FREEZE_PRETRAINED_EMBS_OPTIONS[@]}"; do
  for model_dir in "${!model_epoch_map[@]}"; do
    # Get the list of epochs for the current model directory
    IFS=' ' read -r -a EPOCHES <<< "${model_epoch_map[$model_dir]}"

    for pretrain_epoch in "${EPOCHES[@]}"; do

      f_name="${pretrain_epoch}.pt"
      embeddings_path="${PRETRAINED_MODELS_ROOT}/${model_dir}/${f_name}"

      experiment_name="barlow_twins__item_id_embed_init_with_pretrained_gnn__bs_${BATCH_SIZE}__${model_dir}__pretrain_epoches_${pretrain_epoch}__freeze_${freeze_pretrained_embs}__epoches_${MAX_EPOCHES}"

      echo ""
      echo "EXPERIMENT: ${experiment_name}"
      echo ""

      PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
        --config-dir conf --config-name barlow_twins_params \
        model_path="models/${experiment_name}.p" \
        logger_name="${experiment_name}" \
        \
        data_module.train_batch_size=${BATCH_SIZE} \
        data_module.train_num_workers=4 \
        data_module.valid_batch_size=64 \
        data_module.valid_num_workers=4 \
        \
        +trainer.checkpoints_every_n_val_epochs=10 \
        +trainer.checkpoint_dirpath="./checkpoints/${experiment_name}" \
        +trainer.checkpoint_filename="\{epoch\}" \
        trainer.max_epochs=${MAX_EPOCHES} \
        \
        hydra.run.dir="hydra_outputs/${experiment_name}" \
        \
        trx_embedding_layers=all_except_item_id \
        +graph_path="data/graphs/weighted" \
        +trx_custom_embeddings=pretrained_graph_item_embedder \
        trx_custom_embeddings.mcc_code.embeddings.f="${embeddings_path}" \
        trx_custom_embeddings.mcc_code.freeze=${freeze_pretrained_embs} \
        +pl_module.seq_encoder.trx_encoder.custom_embeddings=\${trx_custom_embeddings} \
        \
        +additional_artifacts_to_save="[git_commit_hash]" \
        # device="cpu"



      PYTHONPATH=.. python -m ptls_extension_2024_research.pl_inference \
        model_path="models/${experiment_name}.p" \
        embed_file_name="${experiment_name}_embeddings" \
        inference.batch_size=128 \
        --config-dir conf --config-name barlow_twins_params
        # +inference.devices=0 \

    done
  done
done
