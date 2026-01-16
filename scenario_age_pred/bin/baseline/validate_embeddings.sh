# Compare
rm results/scenario_age_coles_baseline.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --config-dir conf/embeddings_validation --config-name embeddings_validation__coles_baselines +workers=10 +total_cpu_count=8 # +local_scheduler=false
