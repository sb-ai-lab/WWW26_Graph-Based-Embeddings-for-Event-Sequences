PYTHONPATH=.. python -m ptls_extension_2024_research.create_similarity_matrix_related_feats \
  --graph_dir_path ./data/graphs/weighted_train_graph_has_test_items \
  --save_normalized_sparse_adj_embs \
  --save_min_max_array \
  --min_max_iterative_batch_size 2046