#!/usr/bin/env bash

python create_aggregated_edges.py \
--dataset_path data/original_format_data/competition_data_final_preprocessed.parquet \
--save_path data/competition_data_edges.parquet