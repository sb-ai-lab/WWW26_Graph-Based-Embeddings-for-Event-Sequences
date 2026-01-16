from typing import Dict, Iterable, List, Union, Sequence

import numpy as np

from .bin_types import UserBinsList, UserBinsNoOrder
from .id_converter import IdConverterBase
from .similarity_matrix_slice_getter import SimilarityMatrixSliceGetterBase



def assign_users_to_bins(similarities_1d: np.ndarray, min_val: float, max_val: float, n_bins: int) -> np.ndarray:
    bin_size = (max_val - min_val) / n_bins
    eps = 1e-6
    bin_indices = np.floor((similarities_1d - min_val) / (bin_size + eps)).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    return bin_indices



class UserBinsContainerBase:
    def get_batch_clusters_dict(self, ids_array_like: Sequence[int]) -> Dict[int, UserBinsList]:
        raise NotImplementedError


class UserBinsContainer_Precalculated_IterableIterableSet:
    def __init__(self,
                 bins_dict: List[UserBinsNoOrder],
                 user_id_converter: IdConverterBase,
                 raise_on_bin_is_not_a_set: bool = False) -> None:
        self.bins_dict = bins_dict
        self.user_id_converter = user_id_converter
        self.raise_on_bin_is_not_a_set = raise_on_bin_is_not_a_set
    
    def get_batch_clusters_dict(self, ids: Iterable[int]) -> Dict[int, UserBinsList]:
        global_ids = ids
        local_ids_set = set(self.user_id_converter.convert_external_to_internal(global_ids))
        batch_clusters_dict = {}
        for idx in local_ids_set:
            global_idx = self.user_id_converter.convert_internal_to_external(idx)
            batch_clusters_dict[global_idx] = []
            for cluster in self.bins_dict[idx]:
                if not isinstance(cluster, set):
                    if self.raise_on_bin_is_not_a_set:
                        raise ValueError(f"Expected bins to be of type set, got {type(cluster)}")
                    cluster = set(cluster)

                intersection = cluster & local_ids_set
                if len(intersection) == 0:
                    continue
                batch_clusters_dict[global_idx].append(
                    self.user_id_converter.convert_internal_to_external(intersection))
        return batch_clusters_dict
    


class UserBinsContainer_FromSimilarityMatrix(UserBinsContainerBase):
    def __init__(self,
                 user_id_converter: IdConverterBase,
                 similarity_matrix_slice_getter: Union[SimilarityMatrixSliceGetterBase, np.ndarray],
                 n_bins=10) -> None:
        self.user_id_converter = user_id_converter
        self.similarity_matrix_slice_getter = similarity_matrix_slice_getter
        self.n_bins = n_bins

    def get_batch_clusters_dict(self, ids: Iterable[int]) -> Dict[int, List[List[int]]]:
        global_ids = ids
        local_ids_set = set(self.user_id_converter.convert_external_to_internal(global_ids))
        batch_clusters_dict = {}
        for anchor_user_local_id in local_ids_set:
            batch_neighbors_local_ids = np.array(list(local_ids_set - {anchor_user_local_id}))
            all_other_user_similariies = self.similarity_matrix_slice_getter[anchor_user_local_id, batch_neighbors_local_ids]
            min_val, max_val = np.min(all_other_user_similariies), np.max(all_other_user_similariies)

            bin_indices = assign_users_to_bins(all_other_user_similariies,
                                               min_val, max_val, self.n_bins)
            
            bins = [set() for _ in range(self.n_bins)]
            for bin_idx, neighbor_local_id in zip(bin_indices, batch_neighbors_local_ids):
                neighbor_global_id = self.user_id_converter.convert_internal_to_external(neighbor_local_id)
                bins[bin_idx].add(neighbor_global_id)

            bins_no_empty = [list(bin_) for bin_ in bins if bin_]

            anchor_user_global_idx = self.user_id_converter.convert_internal_to_external(anchor_user_local_id)
            batch_clusters_dict[anchor_user_global_idx] = bins_no_empty
                
        return batch_clusters_dict


def range_with_ignore(start: int, stop: int, ignore: int) -> np.ndarray:
    return np.concatenate([
        np.arange(start, ignore),
        np.arange(ignore + 1 , stop),
    ])


class UserBinsContainer_FromSimilarityMatrixAndMinMaxArray(UserBinsContainerBase):
    def __init__(self,
                 user_id_converter: IdConverterBase,
                 similarity_matrix_slice_getter: Union[SimilarityMatrixSliceGetterBase, np.ndarray],
                 min_and_max_similarities: np.array,  # n_users x 2
                 n_bins=10) -> None:
        self.user_id_converter = user_id_converter
        self.similarity_matrix_slice_getter = similarity_matrix_slice_getter
        self.min_and_max_similarities = min_and_max_similarities
        self.n_bins = n_bins
    
    def _get_min_and_max(self, anchor_user_local_id):
        return tuple(self.min_and_max_similarities[anchor_user_local_id])

    def get_batch_clusters_dict(self, ids: Iterable[int]) -> Dict[int, List[List[int]]]:
        global_ids = ids
        local_ids_set = set(self.user_id_converter.convert_external_to_internal(global_ids))
        batch_size = len(local_ids_set)

        idx_in_slice_to_local_id = np.array(sorted(local_ids_set))

        idxs = np.ix_(idx_in_slice_to_local_id, idx_in_slice_to_local_id)
        # Slicing a `similarity_matrix_slice_getter`` might be costly. 
        # Thus we do the sclicing once (may help due to vectorization).
        similarity_matrix_slice__all_batch_users: np.ndarray = self.similarity_matrix_slice_getter[idxs]  # shape = batch_size x batch_size
        
        batch_clusters_dict = {}
        for anchor_user_idx_in_slice, anchor_user_local_id in enumerate(idx_in_slice_to_local_id):
            min_val, max_val = self._get_min_and_max(anchor_user_local_id)
            
            neighbor_idxs_in_slice = range_with_ignore(0, batch_size, anchor_user_idx_in_slice)
            similarity_matrix_slice__anchor_user_row = similarity_matrix_slice__all_batch_users[anchor_user_idx_in_slice, neighbor_idxs_in_slice]
            bin_indices = assign_users_to_bins(similarity_matrix_slice__anchor_user_row,
                                               min_val, max_val, self.n_bins)
            
            bins = [set() for _ in range(self.n_bins)]
            for bin_idx, neighbor_idx_in_slice in zip(bin_indices, neighbor_idxs_in_slice):
                neighbor_local_id = idx_in_slice_to_local_id[neighbor_idx_in_slice]
                neighbor_global_id = self.user_id_converter.convert_internal_to_external(neighbor_local_id)
                bins[bin_idx].add(neighbor_global_id)

            bins_no_empty = [list(bin_) for bin_ in bins if bin_]

            anchor_user_global_idx = self.user_id_converter.convert_internal_to_external(anchor_user_local_id)
            batch_clusters_dict[anchor_user_global_idx] = bins_no_empty
                
        return batch_clusters_dict
