# Compare
rm results/scenario_gender_2024_research__pretrained_coles_gnn.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation__pretrained_coles_gnn +workers=10 +total_cpu_count=4
