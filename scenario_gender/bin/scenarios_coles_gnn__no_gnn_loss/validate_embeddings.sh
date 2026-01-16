# Compare
rm results/scenario_gender__coles_gnn__no_gnn_loss.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation__scenarios_coles_gnn__no_gnn_loss +workers=10 +total_cpu_count=4
