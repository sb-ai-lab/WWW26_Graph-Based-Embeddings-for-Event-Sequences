# Get data

```sh
cd scenario_gender

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd scenario_gender
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number


sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```









# Experiments

Судя по коду alpha - это коэф. при link prediction

## gamma alteration

В экспериментах ниже альфа не была реализована в коде. Это означает, что альфа = 1.
Вывод делаю по коммиту `0c9edf21c0caac05df7de9c6a05f165da3657a57`: граф невзвешенный, альфы нет, все скрипты, вида `scenario_coles_gnn__g_{g_value}__{has_orig_str}__{gnn_name}.sh` есть.


 |-scenario_coles_gnn__g_0_9__no_orig_emb__gat.sh
 |-scenario_coles_gnn__g_0_5__no_orig_emb__gat.sh
 |-scenario_coles_gnn__g_0_1__no_orig_emb__gat.sh
 |-scenario_coles_gnn__g_0_9__has_orig_emb__gat.sh
 |-scenario_coles_gnn__g_0_1__has_orig_emb__gat.sh

 |-scenario_coles_gnn__g_0_1__no_orig_emb__graph_sage.sh
 |-scenario_coles_gnn__g_0_5__has_orig_emb__graph_sage.sh
 |-scenario_coles_gnn__g_0_5__no_orig_emb__graph_sage.sh
 |-scenario_coles_gnn__g_0_9__has_orig_emb__graph_sage.sh
 |-scenario_coles_gnn__g_0_1__has_orig_emb__graph_sage.sh
 |-scenario_coles_gnn__g_0_9__no_orig_emb__graph_sage.sh

Кажется, что эти результаты находятся в первой latex таблице, но по ошибке стоит альфа = 0.
Но даже если так часть этих экспериментов отсутствует:
 |-scenario_coles_gnn__g_0_9__no_orig_emb__gat.sh
 |-scenario_coles_gnn__g_0_1__no_orig_emb__gat.sh
 |-scenario_coles_gnn__g_0_9__has_orig_emb__gat.sh

 run_all_coles_gnn.sh  -- это просто все сценарии вида `scenario_coles_gnn__g_{g_value}__{has_orig_str}__graph_sage.sh`



---------------------------------------------------------------------------------------------------------------




 |-scenario_coles_gnn_0_5__dotprod__has_orig_emb_weights.sh
 |-scenario_coles_gnn_0_5__dotprod__has_orig_emb.sh








 |-scenario_coles_with_avg_pool.sh





 |-scenarios_coles_gnn_weighted__g_0_5__a_0_5__requires_grad_false
 |    |-w_prediction_BCE_no_orig.sh
 |    |-run_all.sh
 |    |-lp.sh
 |    |-w_prediction_MSE_no_orig.sh
 |    |-w_prediction_BCE.sh
 |    |-run_all__no_orig.sh
 |    |-w_prediction_MSE.sh


 
 |-run-all-scenarios.sh


 |-scenarios_coles_gnn_weighted__contrastive_gnn
 |    |-w_prediction_BCE_no_orig.sh
 |    |-run_all.sh
 |    |-lp.sh
 |    |-w_prediction_MSE_no_orig.sh
 |    |-w_prediction_BCE.sh
 |    |-run_all__no_orig.sh
 |    |-w_prediction_MSE.sh


 |-scenarios_bpr
 |    |-pretrained_config_generation.ipynb
 |    |-run_all__with_pretrained_gnn.sh
 |    |-run_all.sh
 |    |-README.md
 |    |-validate_embeddings.sh
 |    |-run_all__bins_on_fly.sh
 |    |-validate_embeddings__pretrained_gnn.sh
 |    |-config_generation.ipynb


 |-scenarios_coles_gnn_weighted__g_0_95__a_0_05__contrastive_gnn_gs__60_epoches
 |    |-w_prediction_MSE_no_orig.sh



 |-scenarios_coles_gnn__no_gnn_loss
 |    |-config_creation.ipynb
 |    |-run_all.sh
 |    |-validate_embeddings.sh
 |    |-latex_table_creation.ipynb



 |-scenarios_coles_gnn_pretrained
 |    |-inference_checkpoints.sh
 |    |-get_models_from_checkpoints.sh
 |    |-check_run_model.sh
 |    |-run_coles_gnn_pretrained.sh
 |    |-run_all.sh
 |    |-run_all_unfreeze_after_N_epoches.sh
 |    |-validate_coles_gnn_pretrained__embeddings.sh
 |    |-run_best_model__40_epoches.sh
 |    |-run_all.ipynb
 |    |-coles_gnn_pretrained__config_creation.ipynb
 |    |-run_best_model.sh
 |    |-reproducability_check.ipynb
 |    |-run_all_frozen.sh
 |    |-latex_table_creation.ipynb


 |-run_all_coles_gnn_with_orig_emb_NEW.sh


 |-invalidate_graph_ids_for_impossible_input_ids.sh


 |-scenarios_coles_gnn_weighted__g_0_9__a_0_5__contrastive_gnn
 |    |-w_prediction_BCE_no_orig.sh
 |    |-run_all.sh
 |    |-lp.sh
 |    |-w_prediction_MSE_no_orig.sh
 |    |-w_prediction_BCE.sh
 |    |-run_all__no_orig.sh
 |    |-w_prediction_MSE.sh


 |-scenario_coles_gnn_0_5__mlp__has_orig_emb_weights.sh



 |-scenario_coles.sh



 |-coles_without_training.sh



 |-scenarios_coles_gnn_weighted
 |    |-w_prediction_BCE_no_orig.sh
 |    |-run_all.sh
 |    |-lp.sh
 |    |-w_prediction_MSE_no_orig.sh
 |    |-w_prediction_BCE.sh
 |    |-run_all__no_orig.sh
 |    |-w_prediction_MSE.sh


 
 |-scenarios_coles_gnn_weighted_v2
 |    |-w_prediction_BCE_no_orig.sh
 |    |-run_all.sh
 |    |-lp.sh
 |    |-w_prediction_MSE_no_orig.sh
 |    |-w_prediction_BCE.sh
 |    |-run_all__no_orig.sh
 |    |-w_prediction_MSE.sh
 |    |-convert_ckpt_to_pretrained.sh
 |    |-run_all__v2.sh
