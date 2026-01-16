#!/usr/bin/env bash

PYTHONPATH=.. python make_graph.py \
--data_path data/ \
--trx_file competition_data_edges.parquet \
--orig_col_client_id user_id \
--col_item_id url_host_preprocesed \
--test_ids_path "data/test_ids_file.csv" \
--output_graph_path data/graphs/weighted_train_graph_has_test_items \
--output_train_graph_file "train_graph.bin" \
--output_full_graph_file "full_graph.bin" \
--output_client_id2full_graph_id_file "client_id2full_graph_id.pt" \
--output_item_id2full_graph_id_file "item_id2full_graph_id.pt" \
--output_client_id2train_graph_id_file "client_id2train_graph_id.pt" \
--output_item_id2train_graph_id_file "item_id2train_graph_id.pt" \
--log_file "results/graph_mts.txt" \
--item_map_file_path "data/url_host_preprocesed_mapping.csv" \
--client_map_file_path "data/client_id_map.parquet" \
--use_weights
