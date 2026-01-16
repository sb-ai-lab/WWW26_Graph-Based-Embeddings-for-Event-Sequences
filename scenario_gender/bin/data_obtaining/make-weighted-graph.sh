#!/usr/bin/env bash

PYTHONPATH=.. python make_graph.py \
--data_path data/ \
--trx_file transactions.csv \
--orig_col_client_id customer_id \
--col_item_id "mcc_code" \
--cols_log_norm "amount" \
--test_ids_path "data/test_ids.csv" \
--output_graph_path data/graphs/weighted \
--output_train_graph_file "train_graph.bin" \
--output_full_graph_file "full_graph.bin" \
--output_client_id2full_graph_id_file "client_id2full_graph_id.pt" \
--output_item_id2full_graph_id_file "item_id2full_graph_id.pt" \
--output_client_id2train_graph_id_file "client_id2train_graph_id.pt" \
--output_item_id2train_graph_id_file "item_id2train_graph_id.pt" \
--log_file "results/weighted_graph_gender.txt" \
--item_map_file_path "data/mcc_code_mapping.csv" \
--client_map_file_path "data/client_id_map.parquet" \
--use_weights

# 152 sec with    --print_dataset_info
#  52 sec without --print_dataset_info