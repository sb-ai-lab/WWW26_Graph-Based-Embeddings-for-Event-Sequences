# Compare
rm results/embeddings_validation__bpr_with_pretrained_gnn.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation__bpr_with_pretrained_gnn +workers=8 +total_cpu_count=4 +local_scheduler=false