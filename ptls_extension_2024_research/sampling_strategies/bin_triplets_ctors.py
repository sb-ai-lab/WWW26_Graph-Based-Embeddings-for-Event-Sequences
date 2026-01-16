"""
Bin triplet selector constructors exist because both `user_bins_container` and `bin_separation_strategy`
might need to reference the same object. However, there isn't a straightforward way to
initialize a shared object in Hydra and pass it to both classes. While creating two
identical objects in Hydra would technically work, it would lead to unnecessary memory usage.
"""
from typing import Union

import numpy as np  # For typing.
import scipy

from .bin_triplets import BinTriplets
from .bin_triplets_utils.id_converter import IdConverterBase
from .bin_triplets_utils.user_bins_getters import UserBinsContainer_FromSimilarityMatrix, UserBinsContainer_FromSimilarityMatrixAndMinMaxArray 
from .bin_triplets_utils.bin_separation_strategies import MaxDistanceBinSeparationStrategy
from .bin_triplets_utils.similarity_matrix_slice_getter import SimilarityMatrixSliceGetterBase

def create_bin_triplets__from_similarity_matrix(n_bins: int, 
                                                user_id_converter: IdConverterBase,
                                                similarity_matrix_slice_getter: Union[np.ndarray, SimilarityMatrixSliceGetterBase],
                                                num_triplets_per_anchor_user: int,
                                                min_elements_in_bin: int,
                                                ) -> BinTriplets:
    user_bins_container = UserBinsContainer_FromSimilarityMatrix(
        user_id_converter, similarity_matrix_slice_getter, n_bins=n_bins)
    
    bin_separation_strategy = MaxDistanceBinSeparationStrategy(
        min_elements_in_bin=min_elements_in_bin, 
        similarity_matrix_slice_getter=similarity_matrix_slice_getter, user_id_converter=user_id_converter)
    
    return BinTriplets(user_bins_container, 
                       bin_separation_strategy, 
                       num_triplets_per_anchor_user)


def create_bin_triplets__from_similarity_matrix_and_min_max(n_bins: int, 
                                                                                                            user_id_converter: IdConverterBase,
                                                                                                            similarity_matrix_slice_getter: Union[np.ndarray, SimilarityMatrixSliceGetterBase],
                                                                                                            min_and_max_similarities: np.ndarray,
                                                                                                            num_triplets_per_anchor_user: int,
                                                                                                            min_elements_in_bin: int,
                                                                                                            ) -> BinTriplets:
    user_bins_container = UserBinsContainer_FromSimilarityMatrixAndMinMaxArray(
        user_id_converter, similarity_matrix_slice_getter, min_and_max_similarities = min_and_max_similarities, n_bins=n_bins)
    
    bin_separation_strategy = MaxDistanceBinSeparationStrategy(
        min_elements_in_bin=min_elements_in_bin, 
        similarity_matrix_slice_getter=similarity_matrix_slice_getter, user_id_converter=user_id_converter)
    
    return BinTriplets(user_bins_container, 
                       bin_separation_strategy, 
                       num_triplets_per_anchor_user)
