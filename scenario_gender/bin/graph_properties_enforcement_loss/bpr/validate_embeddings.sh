# Compare
rm results/scenario_gender_bpr.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf --config-name embeddings_validation__bpr +workers=8 +total_cpu_count=4
