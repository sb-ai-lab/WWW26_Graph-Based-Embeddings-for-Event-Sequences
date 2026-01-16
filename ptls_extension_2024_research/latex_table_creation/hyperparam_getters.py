def get_batchsize_from_config(config, experiment_name: str) ->  int:
    return int(config['data_module']['train_batch_size'])

def get_split_count_from_config(config, experiment_name: str) -> int:
    return int(config['data_module']['train_data']['splitter']['split_count'])

def get_convex_loss_alpha_from_config(config, experiment_name) -> float:
    return config['pl_module']['loss']['alpha']

def get_triplets_per_user_from_config(config, experiment_name) -> int:
    return config['pl_module']['loss']['loss2']['triplet_selector']['num_triplets_per_anchor_user']

def get_min_users_in_separated_single_bin_from_config(config, experiment_name) -> int:
    triplet_selector = config['pl_module']['loss']['loss2']['triplet_selector']
    if 'bin_separation_strategy' in triplet_selector:
        return triplet_selector['bin_separation_strategy']['min_elements_in_bin']
    else:
        return triplet_selector['min_elements_in_bin']

