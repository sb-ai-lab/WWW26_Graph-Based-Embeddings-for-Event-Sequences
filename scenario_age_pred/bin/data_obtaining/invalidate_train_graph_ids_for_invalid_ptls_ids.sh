    
PYTHONPATH=.. python -m ptls_extension_2024_research.utils.invalidate_graph_ids_for_impossible_input_ids \
    --orig_client_id_column "client_id" \
    --item_id_column "small_group" \
    --invalid_train_graph_id 500000 \
    \
    --not_encoded_test_client_ids_path ./data/test_ids_file.csv \
    --client_id_map_path ./data/client_id_map.parquet \
    --train_dataset_path ./data/train_trx_file.parquet \
    --orig_client_id2train_graph_id_tensor_path ./data/graphs/weighted/client_id2train_graph_id.pt \
    --orig_item_id2train_graph_id_tensor_path ./data/graphs/weighted/item_id2train_graph_id.pt \
    --out_client_id2train_graph_id_tensor_path ./data/graphs/weighted/client_id2train_graph_id_invalidated.pt \
    --out_item_id2train_graph_id_tensor_path ./data/graphs/weighted/item_id2train_graph_id_invalidated.pt \
    --out_client_id2train_graph_id_dict_path ./data/graphs/weighted/client_id2train_graph_id__dict.pt \
    --out_item_id2train_graph_id_dict_path ./data/graphs/weighted/item_id2train_graph_id__dict.pt \
