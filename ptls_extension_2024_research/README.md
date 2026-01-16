# ptls extensions used in the coles-gnn-experiments research project


**Library modules:**

- `.frames`
    - `.frames.coles_client_id_aware` - CoLES that uses a dataset with real client_ids unlike the original CoLES.
    - `.frames.coles_gnn` - Joint CoLES+Gnn training.
    - `.frames.gnn` - Gnn training (can be used for separate Gnn training; is also used within frames.coles_gnn)
- `.graphs` - Contains graph-and-gnn-related classes. 
    - `.graphs.static_models` - Static GNN models. Given a (sub)-graph, batch of node features, and batch of edge weights, they return the updated node features.
    - `.graphs.graph_construction` - Scripts to construct a graph from a dataset that has events as entries. Has scripts for gender, age and mts datasets.
    -  `graphs.graph` - classes that store a graph and can return a subgraph given the nodes.
    - `.graphs.utils` - link-prediction head options, rand edge sampler, `create_subgraph_with_all_neighbors` function
- `latex_table_creation` - a set of functions used to create latex tables from saved hydra-config and results.txt. Note: all age and mts experiments save hydra configs with same name as model `.p` file, embeddings `.pickle` file and entry in results `.txt` file to be able to know for sure the hyperparameters and as a bonus to automatically generate latex tables from those artifacts reducing the chance of human error. 
- `.additional_callbacks` - additional pl callbacks used in `.pl_train_module`. Primarily used to freeze/unfreeze pretrained gnn embeddings during training.
- `.nn.seq_encoder` - merely AvgPoolLinearSeqEncoder
- `.nn.trx_encoder.encoders` - custom_encoders for a trx_encoder.custom_embeddings
- `.losses` - BPRLoss and ConvexCombinationLoss (a convex combination of given two losses)
- `.sampling_strategies` - triplet_selectors. Contains triplet selector based on bins that is used in BPRLoss.
- `.utils` - contains 1) a script to convert a checkpoint to a model (used in pl_inference); 2) invalidate_graph_ids_for_impossible_input_ids - script to update ptls_ids_to_graph_ids maps by setting a value for ids outside train set to a value grater than `number_of_graph_nodes` (so that impossible indexes lead to indexation error).
- `.hydra_utils` - wrappers for functions that are hard to use otherwise as `_target_` fields in hydra config. 
- `.lightning_utils` - contains `LogLstEl`. It is as an entry in a list of items to log at current step. Is needed when a pl_module has other pl_modules as attributes and we want to log something from them. a `self.log` cannot be used inside the inner pl_module in this case. 
- `.make_datasets_spark` - differences from `ptls.make_datasets_spark`: cols_event_time has "#mts"; can save `original_categorical_feats -> encoded` map; Can encode client_ids and save `original_client_ids -> encoded` map; has `cols_no_transform`, `cols_to_float`. Saving maps is useful for creating graphs.
- `.pl_train_module` - extends `ptls.pl_train_module` by adding extra possible config keys: `additional_callbacks`, `additional_artifacts_to_save`, `model_weights_only_ckpt`, `trainer.resume_checkpoint_path`.
- `.pl_inference` - differs from `ptls.pl_inference` by fixing a `accelerator=cpu leads to trainer initialization error` bug. Original version if config has accelerator=cpu passes `accelerator='cpu', devices=0` to the trainer. But `devices=0 is not a valid input using cpu accelerator`. Thus we pass `devices=auto` instead.
- `.pl_train_module_utils` - currently only `get_git_commit_hash` function. Used in `pl_train_module` to save commit hash in `additional_artifacts` for reproducibility.
- `.nn.seq_encoder.containers.AvgPoolLinearSeqEncoder` - A seq_encoder that averages the embeddings of the sequence and passes it through a linear layer. Was used at the start of a research to confirm the importance of the sequential structure.  



**Legacy modules:**

Legacy modules are kept for reproducibility and are not used in the most recent experiments. However, they may be useful for future research. 

> [!NOTE]  
> Most of the legacy models are related to including an actual GNN model as a custom embedder. In the current setting, coles stores the gnn_embeddings  (attached to GNN's computational graph in the case of joint training) and a ColesGNNModule instance just updates theese embeddings during training. 


- `.nn.trx_encoder.trx_encoder_with_client_item_embeddings.TrxEncoder_WithCIEmbeddings` - trx_encoder that supports client_item_embeddings. 
- `.nn.trx_encoder.client_item_encoder` - contains StaticGNNTrainableClientItemEncoder (a GNN model used as embedder and a coles_batch_to_subgraph_converter). Was used in `TrxEncoder_WithCIEmbeddings`.
- `.pl_inference_with_client_ids` - `ptls.pl_inference` but uses `InferenceModuleClientIdAware` instead of `InferenceModule`. Is used to support inference for seq_encoders that use `TrxEncoder_WithCIEmbeddings` (and a dataset that provides client_ids alongside with `PaddedBatch`).







# Main components:


## TrxEncoder_WithCIEmbeddings
* location: `nn.trx_encoder.trx_encoder_with_client_item_embeddings.TrxEncoder_WithCIEmbeddings`

> [!NOTE]  
> TrxEncoder_WithCIEmbeddings is not used in recent experiments. However, it can be helpful in future research. One potential use case where this class would be needed is incorporating GNN _edge_ embeddings as client-item embeddings. 
>
> TrxEncoder_WithCIEmbeddings is nesessary in case of including an actual GNN as a custom embedder, but in the current setting, coles stores the gnn_embeddings (attached to the GNN's computational graph in the case of joint training).


Extends `TrxEncoder` from ptls by a client_item_embeddings property. `trx_encoder_instance.client_item_embeddings` is an an instance of `BaseClientItemEncoder`'s child. `BaseClientItemEncoder`s take a client_ids tensor of shape `(batch_size,)` and an item_ids tensor of shape `(batch_size, seq_len)` as input and returns torch.tensor of shape `(batch_size, seq_len, client_item_embedding_size)`. 

Another technical difference from `TrxEncoder` is that input is not just `PaddedBatch` of sequential transaction features but a `Tuple[PaddedBatch, torch.Tensor]` where the tensor is a batch of client_ids. 

In our research client-item embeddings are GNN embeddings of items in a user-item graph. However we can use original TrxEncoder if we don't only store


## CoLESDataset
* location: `.frames.coles_client_id_aware.coles_dataset_real_client_ids.CoLESDataset`

Same as `ColesDataset` from ptls, but 
* Collate_fn returns REAL client_ids instead of 
    just different integers for different clients retirieved via enumerate.
* An i-th dataset element contains not only n dicts representing 
    splits of sequential features, but also an id. This is required
    to get real ids in collate_fn.

## CoLESModule_CITrx
* location: `.frames.coles_client_id_aware.coles_module__trx_with_ci_embs.CoLESModule_CITrx`

Similar to as `CoLESModule` from ptls, except it's expected that our version of `ColesDataset` is used and trx_encoder is our `TrxEncoder_WithCIEmbeddings` instead of `TrxEncoder`. This means that real client_ids should be provided in the dataset and trx_encoder should take `Tuple[PaddedBatch, torch.Tensor]` and be able to create client-item embeddings.

## make_datasets_spark
* location: `.make_datasets_spark`

Differences from `ptls.make_datasets_spark`:
* -cols_event_time has "#mts" option. 
    <!--Format is `#mts {col_date} {col_part_of_day}. The resulting time value is $\frac{date\_value.to\_unix\_timestamp()}{number\_of\_seconds\_in\_day} + part\_of\_day\_map[part\_of\_day\_value]$ where _part_of_day_map = `{'day': 0.0, 'morning': 0.25, 'evening': 0.5, 'night': 0.75}`.-->
* Can save `original_categorical_feats -> encoded` map
* Can encode client_ids and save `original_client_ids -> encoded` map
* Has `cols_no_transform`, `cols_to_float`

Saving maps is useful for creating graphs.

`.make_datasets_spark.DatasetConverter` will prbably become a child of `ptls.make_datasets_spark.DatasetConverter`


# Sampling Strategies
* location: `.sampling_strategies`

A sampling strategy is responsible for selecting embeddings from a batch, given 
an `embeddings` tensor of all embeddings and their corresponding labels (`client_ids`). 
It is important to note that a single user has multiple embeddings in the same batch.

Triplet selectors produce tensors of triplets, where each triplet consists of:
- `anchor_user_embed_id`: The index of the anchor embedding.
- `positive_user_embed_id`: The index of the positive embedding (similar to the anchor).
- `negative_user_embed_id`: The index of the negative embedding (dissimilar to the anchor).

All of these indices are from `embeddings` tensor mentioned above.

## BinTriplets

Bins are disjoint subsets of users from the batch, grouped according to their similarity to the anchor user. 
All users within a bin have close values of similarity to relative to achor_user. 
The union of all bins includes all users in the batch, excluding the anchor user.
For each achor_user all their bins are stored in a list, sorted by similarity to the anchor user.

`BinTriplets` is a triplet selector that, for each `anchor_user`, selects:
- A `positive_user` from a bin that contains users with higher similarity to the anchor.
- A `negative_user` from a bin that contains users with lower similarity to the anchor.

These bins are generated by a child class of `UserBinsContainerBase`, which is an abstraction 
designed to return the appropriate bins for each `anchor_user` given all user IDs present in the batch.

### BinTriplets in Our Research

In our research, `BinTriplets` is used as a triplet selector in contrastive loss functions 
such as `BPRLoss` (Bayesian Personalized Ranking Loss), with the goal of preserving 
structural proximity between node representing users in a bipartite users-items graph. 
In other words we use a contrastive loss with `BinTriplets` to ensure that embeddings 
formed by `CoLES` (or other model) are endowed with properties of the nodes in the graph.

The `UserBinsContainer` implementations developed for this research generate bins based on cosine similarity 
between user features derived from adjacency matrix. These features are vectors of length `n_items`, 
where each element represents the weight of a graph edge connecting the user to an item (with zeros indicating no connection).

A similarity matrix of shape `n_users x n_users` is formed via computing cosine similarities between such features.
The rows of the matrix (corresponding to a user) is used to divide users into `n_bins` bins of equal width.

### Recomendations on `UserBinsContainer` choice

There are three UserBinsContainer implementations:
1. `UserBinsContainer_Precalculated_IterableIterableSet`
    * Applicable on very small datasets (ex. gender) due to extensive RAM usage
    * Uses precalculated bins
2. `UserBinsContainer_FromSimilarityMatrix`
    * Has the same asimptotic complexity as `UserBinsContainer_Precalculated_IterableIterableSet`
        but is applicable with bigger datasets (ex. age)
    * Is used with an actual similarity matrix (though `SimilarityMatrixSliceGetter` is supported) 
    * Computes bins on-the-go.
3. `UserBinsContainer_FromSimilarityMatrixAndMinMaxArray`
    * The only option for big datasets (ex. mts). It's highly RAM efficient
        and has a compatible execution time to other implementations
    * Is used with `min_max_array` and `SimilarityMatrixSliceGetter`
    * `SimilarityMatrixSliceGetter` has interface of an actual matrix (np.array),
        but computes the indexed fragment of the similarity matrix 
        instead of storing the matrix
    * `SimilarityMatrixSliceGetter` is highly RAM efficient, but needs 
        to perform some computations each time data is retrieved (in case 
        of `SimilarityMatrixSliceGetter__FromFeatsBeforeDotProduct`
        the "computations" is a dot product). The `UserBinsContainer_FromSimilarityMatrixAndMinMaxArray`
        minimizes the number of times we retrieve data from `SimilarityMatrixSliceGetter`
        making this `UserBinsContainer` implementation highly RAM efficient 
        and of a compatible execution time to other implementations
    * If the dataset is small  `SimilarityMatrixSliceGetter` can be replaced with 
        actual similarity_matrix

