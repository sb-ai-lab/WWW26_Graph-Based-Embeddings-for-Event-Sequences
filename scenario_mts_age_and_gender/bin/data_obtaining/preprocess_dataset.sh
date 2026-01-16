#!/usr/bin/env bash

python preprocess_dataset.py \
--dataset_path data/original_format_data/competition_data_final.parquet \
--save_path data/original_format_data/competition_data_final_preprocessed.parquet