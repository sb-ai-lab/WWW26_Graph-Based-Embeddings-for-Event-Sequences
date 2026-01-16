#!/bin/bash

REPORT_FILE_NAME="scenario_age__barlow_twins_bpr.txt"
CONFIG_DIR="conf/embeddings_validation"
CONFIG_NAME="embeddings_validation_barlow_twins_bpr"

rm "results/${REPORT_FILE_NAME}"

PYTHONPATH=.. python -m ptls_extension_2024_research.utils.update_validation_config \
    --config-dir conf/update_validation_config --config-name config \
    output_config_path="./${CONFIG_DIR}/${CONFIG_NAME}.yaml" \
    filter=regex \
    filter.pattern="barlow_twins_bpr"

# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir ${CONFIG_DIR} --config-name ${CONFIG_NAME} +workers=24 +total_cpu_count=24 report_file_name=${REPORT_FILE_NAME}  # +local_scheduler=false
