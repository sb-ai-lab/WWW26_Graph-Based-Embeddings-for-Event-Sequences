MAX_EPOCHES=150

# * Batch size 64 is over 2 Gb
# * Batch size 48 is bad because last batch is too small.
#   and we cannot sample 5 negative examples (neg_count in HardNegativeMining)
# * Batch size 46 is sometimes ok for 2Gb gpu, but once CUDA went out of memory. It was just 1 Mb short
BATCH_SIZE=40
SPLIT_COUNT=2


MODEL_NAME="coles__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__${MAX_EPOCHES}_epoches"
model_path="models/${MODEL_NAME}_model.p"

PYTHONPATH=.. python -m ptls_extension_2024_research.pl_train_module \
    --config-dir conf --config-name mles_params \
    data_module.train_data.splitter.split_count=${SPLIT_COUNT} \
    data_module.valid_data.splitter.split_count=${SPLIT_COUNT} \
    pl_module.validation_metric.K=1 \
    pl_module.lr_scheduler_partial.step_size=60 \
    model_path="${model_path}" \
    logger_name="${MODEL_NAME}"  \
    data_module.train_batch_size=${BATCH_SIZE} \
    data_module.train_num_workers=4 \
    data_module.valid_batch_size=64 \
    data_module.valid_num_workers=4 \
    \
    +trainer.checkpoints_every_n_val_epochs=10 \
    +trainer.checkpoint_dirpath="./checkpoints/${MODEL_NAME}" \
    +trainer.checkpoint_filename="\{epoch\}" \
    trainer.max_epochs="${MAX_EPOCHES}" \
    \
    hydra.run.dir="hydra_outputs/${MODEL_NAME}" \



# ! Even batch_size 50 takes over 2 Gb on inference  !
python -m ptls.pl_inference    \
    model_path="${model_path}" \
    embed_file_name="${MODEL_NAME}_embeddings" \
    inference.batch_size=45 \
    --config-dir conf --config-name mles_params 
    