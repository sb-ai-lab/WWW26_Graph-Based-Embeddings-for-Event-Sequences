target_name=gender
report_file_name="mts_${target_name}_embedding_validation.txt"

PYTHONPATH=.. python -m ptls_extension_2024_research.utils.update_validation_config \
    --config-dir conf/update_validation_config --config-name config

rm "results/${report_file_name}"
# rm -r conf/embeddings_validation_${target_name}.work/
python -m embeddings_validation \
    --config-dir conf/embeddings_validation --config-name embeddings_validation +workers=8 +total_cpu_count=8 target_name=${target_name} report_file_name=${report_file_name} target.col_target="is_male" models/lgbm=${target_name} # +local_scheduler=false
