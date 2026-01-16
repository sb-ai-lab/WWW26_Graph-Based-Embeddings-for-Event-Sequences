from typing import Iterable, Dict, List
from collections import defaultdict
import random

import torch

from .triplet_selector import TripletSelector
from .bin_triplets_utils.user_bins_getters import UserBinsContainerBase
from .bin_triplets_utils.bin_separation_strategies import BinSeparationStrategy


def get_two_differenct_random_numbers(max_val):
    a = torch.randint(0, max_val, (1,)).item()
    b = torch.randint(0, max_val, (1,)).item()
    while a == b:
        b = torch.randint(0, max_val, (1,)).item()
    return a, b


class BinTriplets(TripletSelector):
    """
    BinTriplets is a triplet selector that iterates over `anchor_user_id`'s 
    (unique user IDs in the batch) and, for each `anchor_user_id`, 
    samples a `positive_user_id` and a `negative_user_id`. 
    For each user ID in the triplet, a random embedding of that user is 
    sampled from the batch.

    Positive and negative users are sampled from **bins**, which are 
    disjoint subsets of users from the batch that have close values of 
    similarity to the anchor user. These bins are stored in a list, sorted 
    by similarity to the `anchor_user`. A positive user is 
    sampled from a bin with higher similarity than the negative user.

    Properties:
    -----------
    user_bins_container: UserBinsContainerBase
        Responsible for organizing users into similarity-based bins for each anchor user.
    bin_separation_strategy: BinSeparationStrategy
        Strategy used to separate users into different bins if there is only one bin 
        for an anchor user. This ensures that there are at least 
        two bins to sample from.
    num_triplets_per_anchor_user: int
        Specifies the number of triplets to generate for each anchor user.
    """
    def __init__(self, 
                 user_bins_container: UserBinsContainerBase,
                 bin_separation_strategy: BinSeparationStrategy,
                 num_triplets_per_anchor_user: int) -> None:
        self.user_bins_container = user_bins_container
        self.num_triplets_per_anchor_user = num_triplets_per_anchor_user
        self.bin_separation_strategy = bin_separation_strategy
    
    def get_two_differenct_random_numbers(self, max_val):
        return get_two_differenct_random_numbers(max_val)
    
    def _get_id_2_idxs(self, ids_np: Iterable[int]) -> Dict[int, List[int]]:
        """
        Creates a mapping from user_id to a list containing indices 
        for all their embeddings in the batch.
        """
        id_2_idxs = defaultdict(list)
        for idx, id_ in enumerate(ids_np):
            id_2_idxs[id_].append(idx)
        return id_2_idxs

    def get_triplets(self, _, ids):
        ids_np = set(ids.detach().cpu().numpy())
        id_2_idxs = self._get_id_2_idxs(ids_np)
        ids_set = set(ids_np)

        batch_clusters_dict = self.user_bins_container.get_batch_clusters_dict(ids_set)
        triplets = []
        for anchor_user_id in ids_set:
            clusters = batch_clusters_dict[anchor_user_id]
            if len(clusters) == 1:
                clusters = self.bin_separation_strategy(
                    user_ids_bin = clusters[0],
                    anchor_user_id = anchor_user_id
                )
            if clusters is None:
                # `bin_separation_strategy` decided that the cluster is invalid (impossible to separate).
                continue
            for _ in range(self.num_triplets_per_anchor_user):
                negative_cluster_idx, positive_cluster_idx = sorted(self.get_two_differenct_random_numbers(len(clusters)))
                positive_cluster, negative_cluster = clusters[positive_cluster_idx], clusters[negative_cluster_idx]
                positive_user_id = positive_cluster[torch.randint(0, len(positive_cluster), (1,)).item()]
                negative_user_id = negative_cluster[torch.randint(0, len(negative_cluster), (1,)).item()]
                triplets.append([anchor_user_id, positive_user_id, negative_user_id])
        

        triplets_embed_idxs = []
        for ids_triplet in triplets:
            triplets_embed_idxs.append(
                [random.choice(id_2_idxs[id_]) for id_ in ids_triplet]
            )

        return torch.tensor(triplets_embed_idxs)
