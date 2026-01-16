# Experiments for the "Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models" paper


The basic documentation for all modules created to extend pytorch-lifestream library and allow us to perform the experiments can be found [here](./ptls_extension_2024_research/README.md)



# Setup and test using pipenv

```sh
# Ubuntu 20.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

# run luigi server
luigid
# check embedding validation progress at `http://localhost:8082/`

# use tensorboard for metrics exploration
tensorboard --logdir lightning_logs/ 
# check tensorboard metrics at `http://localhost:6006/`

```

# Run scenario
 We check 3 datasets as separate experiments. See `README.md` files in experiments folder:
 - [Age](scenario_age_pred/README.md)
 - [MTS](./scenario_mts_age_and_gender/README.md)
 - [Gender](scenario_gender/README.md)

We also check the performance of the model on a close-sourced internal dataset

## Common scripts 
* bin/get-data.sh downloads original data. It's usually a table with each row representing a single transaction
* bin/make_datasets_spark_file.sh uses spark to convert the dataset to a ptls format. The result are 3 files: `train_trx_file.parquet`, `test_trx_file.parquet` and `test_ids_file.csv` and optionally files contating map maps old_value -> new_value for categorical features and user_id. The train_trx_file.parquetand test_trx_file.parquet are parquet files where rows are users and for each row the column (the features) are sequences represented as 1d numpy arrays. 


# Conduct auxiliary loss experiment on your dataset
To conduct the `auxiliary loss` experiments described in the paper on your custom dataset you can reference the code in ./scenario_age as the default template

Configs you'll need:
1. Copy all content of scenario_age_pred/conf/loss. It contains all configs for auxiliary bpr and triplet losses. You can leave everything as is there
2. Copy to your conf dir `scenario_age_pred/conf/mles_params_client_id_aware` and all configs that it relies on:  `dataset_unsupervised`, `inference`, `trx_embedding_layers`, `data_module` 

What you’ll need to change in the configs:
* Seems like the only things to change are the features: all content of `conf/trx_embedding_layers/all_features.yaml` should be replaced with a list of your categorical features and in `conf/mles_params_client_id_aware` `numeric_values` should be filled with your numeric features

Scripts you’ll need
1. Copy `scenario_age_pred/bin/graph_properties_enforcement_loss` to the bin dir of your scenario
2. Copy `scenario_age_pred/bin/data_obtaining/create_similarity_matrix_related_feats.sh`

You don’t have to change anything in the scripts, but you may want to try other hyperparameters in bin/graph_properties_enforcement_loss/bpr/run_all.sh and bin/graph_properties_enforcement_loss/triplet/run_all.sh  (all hyperparameters are defined at the beginning of the scripts).
Note that if you change hyperparameters in bin/graph_properties_enforcement_loss/bpr/run_all.sh you will need to make the same changes in get_models_from_checkpoints.sh

Run bin/data_obtaining/create_similarity_matrix_related_feats.sh ⇒ `min_max_array.npy` and `graph_adj_embs_normalized.npz` should appear in `YOUR_SCENARIO_NAME/data/graphs/weighted`.

Run bin/graph_properties_enforcement_loss/bpr/run_all.sh

Run bin/graph_properties_enforcement_loss/triplet/run_all.sh

When the training is finished you can collect models from checkpoints via get_models_from_checkpoints.sh and then perform embedding validation via validate_embeddings.sh



# Notebooks

Full scenarious are console scripts configured by hydra yaml configs.

# Results

All results are stored in `*/results` folder.

# Notes

* Docker-related instrucitons can be found [here](./notes/docker.md). The docker image contains the dependencies to run the experiments. However, docker was not used for the most of the experiments due to the restrictions on our computing cluster