# Compare
rm results/coles__item_id_embedding_init_with_gnn_embs.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf/embeddings_validation --config-name embeddings_validation__coles__item_id_embedding_init_with_gnn_embs +workers=10 +total_cpu_count=8 +local_scheduler=false
