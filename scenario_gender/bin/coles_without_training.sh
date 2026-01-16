# # Check COLEs with split_count=2
# python -m ptls.pl_train_module \

MAX_EPOCHES=0
BATCH_SIZE=64
SPLIT_COUNT=2

MODEL_NAME="coles__batch_size_${BATCH_SIZE}__split_count_${SPLIT_COUNT}__${MAX_EPOCHES}_epoches"
model_path="models/${MODEL_NAME}_model.p"

python -m ptls.pl_train_module \
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
    data_module.valid_num_workers=4  \
    trainer.max_epochs=${MAX_EPOCHES}  \
    \
    hydra.run.dir="hydra_outputs/${MODEL_NAME}" \
    \
    device=cpu \


# ! batch_size 50 takes over 2 Gb on inference  !
PYTHONPATH=..  python -m ptls_extension_2024_research.pl_inference    \
    model_path="${model_path}" \
    embed_file_name="${MODEL_NAME}_embeddings" \
    inference.batch_size=16 \
    +inference.devices=0 \
    --config-dir conf --config-name mles_params


# Compare
rm results/escenario_gender_no_training.txt
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation__no_training +workers=8 +total_cpu_count=4