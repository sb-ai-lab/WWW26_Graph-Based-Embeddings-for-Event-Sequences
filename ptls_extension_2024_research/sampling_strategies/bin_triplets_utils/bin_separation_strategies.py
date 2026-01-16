"""
Bin separation strategy is a callable used to separate a bin 
into two when there is only one bin present for a user and we need atleast two.
"""
from typing import Tuple, Iterable, List, Union, Sequence

import numpy as np
import torch

from .id_converter import IdConverterBase
from .similarity_matrix_slice_getter import SimilarityMatrixSliceGetterBase



def get_max_distance_idx__all_search_space_is_const(sorted_arr: Sequence[float], start: int, end: int) -> float:
    """
    Returns max distance index in case when all elements of the search space are equal.

    Expands the search space on both sides (or one side if expending the other is impossible) 
    up the to nearest element that differs from the constant.
    """
    assert sorted_arr[start-1] == sorted_arr[end-1]

    const_val = sorted_arr[start - 1]

    while start > 1 and sorted_arr[start] == const_val:
        start -= 1

    while end < len(sorted_arr) and sorted_arr[end - 1] == const_val:
        end += 1

    if const_val - sorted_arr[start-1] > sorted_arr[end - 1] - const_val:
        return start
    else:
        return end - 1 
    


# Note that for smaller sorted_arr of type list (e.x. size=40) 
# get_max_distance_idx__sequence(sorted_arr) is faster then get_max_distance_idx__numpy(np.array(sorted_arr)).
def get_max_distance_idx__sequence(sorted_arr: Sequence[float], margin: int) -> int: 
    """
    Finds the index of the maximum difference between consecutive elements in a sorted sequence of float.
    Works for any sequence of floats (tuple, list, np.ndarra, torch.tensor etc.) 
    Has a specified margin to ensure that there are at least `margin` elements before `max_diff_index`
    and atleast `margin - 1` elements after `max_diff_index`.

    The function is used to split one bin into two. `max_diff_index` can be thought of 
    as the first element of the second bin, and `margin` can be thout of as the minimum elements per bin.

    Arguments:
    ----------
    sorted_arr: List[float]
        A list of sorted numerical values.
    margin: int
        Minimal allowed `max_diff_index` is `margin+1`. Maximum allowed `max_diff_index` is `len(sorted_arr) - margin`.
        If all elements in range(margin, len(sorted_arr) - margin + 1) are equal, the search space is extended 
        on both sides up to the nearest different elements.

    Returns:
    --------
    int: The index of the element where the maximum difference occurs (in relation to the previous element).
    """
    # margin 0 makes no sense in context of separating one bin into two.
    # However if we had margin=0 allowed, len(sorted_arr) - margin + 1 should have been 
    # replaced with min(len(sorted_arr), len(sorted_arr) - margin + 1).
    assert margin >= 1

    if sorted_arr[0] == sorted_arr[-1]:  # All elements are equal.
        return None
    
    # In case len(sorted_arr) == margin*2 we would iterate over an empty sequence  
    if len(sorted_arr) == margin*2:
        return margin
    
    start = margin
    end = len(sorted_arr) - margin + 1

    if sorted_arr[start-1] == sorted_arr[end-1]:
        return get_max_distance_idx__all_search_space_is_const(sorted_arr, start, end)


    max_diff = float('-inf')
    max_diff_index = -1

    for i in range(start, end):
        diff = sorted_arr[i] - sorted_arr[i - 1]

        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    return max_diff_index


def get_max_distance_idx__tensor(sorted_arr: torch.Tensor, margin: int) -> int:
    """
    Finds the index of the maximum difference between consecutive elements in a sorted tensor.
    Has a specified margin to ensure that there are at least `margin` elements before `max_diff_index`
    and at least `margin - 1` elements after `max_diff_index`.

    The function is used to split one bin into two. `max_diff_index` can be thought of 
    as the first element of the second bin, and `margin` can be thought of as the minimum elements per bin.

    Arguments:
    ----------
    sorted_arr: torch.Tensor
        A 1D tensor of sorted numerical values.
    margin: int
        Minimal allowed `max_diff_index` is `margin+1`. Maximum allowed `max_diff_index` is `len(sorted_arr) - margin`.
        If all elements in range(margin, len(sorted_arr) - margin + 1) are equal, the search space is extended
        on both sides up to the nearest different elements.

    Returns:
    --------
    int: The index of the element where the maximum difference occurs (in relation to the previous element).
    """
    assert margin >= 1

    if sorted_arr[0] == sorted_arr[-1]:  # All elements are equal.
        return None
    
    # In case len(sorted_arr) == margin*2 we would select an empty sequence  
    if len(sorted_arr) == margin*2:
        return margin
    
    start = margin
    end = len(sorted_arr) - margin + 1

    if sorted_arr[start-1] == sorted_arr[end-1]:
        return get_max_distance_idx__all_search_space_is_const(sorted_arr, start, end)

    diffs = sorted_arr[1:] - sorted_arr[:-1]  # diffs[i] = sorted_arr[i+1] - sorted_arr[i]
    return torch.argmax(diffs[margin - 1:len(diffs) - margin + 1]) + margin



# Note that for smaller sorted_arr of type list (e.x. size=40) 
# get_max_distance_idx__sequence(sorted_arr) is faster then get_max_distance_idx__numpy(np.array(sorted_arr)).
def get_max_distance_idx__numpy(sorted_arr: np.ndarray, margin: int) -> int:
    """
    Finds the index of the maximum difference between consecutive elements in a sorted NumPy array.
    Has a specified margin to ensure that there are at least `margin` elements before `max_diff_index`
    and at least `margin - 1` elements after `max_diff_index`.

    The function is used to split one bin into two. `max_diff_index` can be thought of 
    as the first element of the second bin, and `margin` can be thought of as the minimum elements per bin.

    Arguments:
    ----------
    sorted_arr: np.ndarray
        A 1D NumPy array of sorted numerical values.
    margin: int
        Minimal allowed `max_diff_index` is `margin+1`. Maximum allowed `max_diff_index` is `len(sorted_arr) - margin`.
        If all elements in range(margin, len(sorted_arr) - margin + 1) are equal, the search space is extended
        on both sides up to the nearest different elements.

    Returns:
    --------
    int: The index of the element where the maximum difference occurs (in relation to the previous element).
    """
    assert margin >= 1

    if sorted_arr[0] == sorted_arr[-1]:  # All elements are equal.
        return None
    
    # In case len(sorted_arr) == margin*2 we would select an empty sequence  
    if len(sorted_arr) == margin*2:
        return margin
    
    start = margin
    end = len(sorted_arr) - margin + 1

    if sorted_arr[start-1] == sorted_arr[end-1]:
        return get_max_distance_idx__all_search_space_is_const(sorted_arr, start, end)

    diffs = sorted_arr[1:] - sorted_arr[:-1]  # diffs[i] = sorted_arr[i+1] - sorted_arr[i]
    return np.argmax(diffs[margin - 1: len(diffs) - margin + 1]) + margin




def sparate_bin_on_max_distance(user_ids_bin: Iterable[int], 
                                similarities: np.ndarray, 
                                min_elements_in_bin: int):
    """
    Separates a bin of user IDs into two bins by finding the index of the maximum difference in their similarities.

    Arguments:
    ----------
    user_ids_bin: Iterable[int]
        Bin of users to be separated into two. Represented as an iterable of user IDs.  
    similarities: np.ndarray
        An array of `float`. similarities[i] is similarity between anchor_user and the user with id user_ids_bin[i].
    min_elements_in_bin: int
        Minimum number of elements required in each resulting bin.

    Returns:
    --------
    Tuple[list[int], list[int]]: Two bins separated based on the maximum similarity difference.

    Raises:
    -------
    AssertionError (Postcondition): If either resulting bin has fewer elements than `min_elements_in_bin`.
    """
    # sorted_similarities, indices = zip(
    #     *sorted((sim, idx) for idx, sim in enumerate(similarities)))
    indices = np.argsort(similarities)
    sorted_similarities = similarities[indices]

    max_diff_idx = get_max_distance_idx__numpy(sorted_similarities, min_elements_in_bin)

    bin_1_idxs, bin_2_idxs = indices[:max_diff_idx], indices[max_diff_idx:]

    user_ids_bin_1 = [user_ids_bin[idx] for idx in bin_1_idxs]
    user_ids_bin_2 = [user_ids_bin[idx] for idx in bin_2_idxs]

    # Theese postconditions are only applicable if search scope was not extended in get_max_distance_idx.
    # assert len(user_ids_bin_1) >= min_elements_in_bin, f"Postcondition is not met: min_elements_in_bin = {min_elements_in_bin}; len(user_ids_bin_1) = {len(user_ids_bin_1)}; initial bin len = {len(user_ids_bin)}"
    # assert len(user_ids_bin_2) >= min_elements_in_bin, f"Postcondition is not met: min_elements_in_bin = {min_elements_in_bin}; len(user_ids_bin_1) = {len(user_ids_bin_2)}; initial bin len = {len(user_ids_bin)}"

    return user_ids_bin_1, user_ids_bin_2



class BinSeparationStrategy:
    """
    Bin separation strategy is a callable used to separate a bin 
    into two when there is only one bin present for a user and we need atleast two.
    """
    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class RaiseErrorSeparationStrategy(BinSeparationStrategy):
    def __init__(self) -> None:
        pass
    
    def __call__(self, *args, **kwargs):
        raise ValueError("More then one bin is present!! \n" \
                         "RaiseErrorSeparationStrategy is a dummy separation " \
                         "strategy used when it's guaranteed that each user has atleast two bins.")    


class MaxDistanceBinSeparationStrategy(BinSeparationStrategy):
    def __init__(self,
                 min_elements_in_bin: int, 
                 similarity_matrix_slice_getter: Union[SimilarityMatrixSliceGetterBase, np.ndarray],
                 user_id_converter: IdConverterBase) -> None:
        self.min_elements_in_bin = min_elements_in_bin
        self.similarity_matrix_slice_getter = similarity_matrix_slice_getter
        self.user_id_converter = user_id_converter

    def get_similarities(self, anchor_user_id: int, bin_with_ids: List[int]) -> np.ndarray:
        anchor_user_external_id = anchor_user_id
        bin_with_external_ids = bin_with_ids

        achor_user_internal_id = self.user_id_converter.convert_external_to_internal(anchor_user_external_id)
        bin_with_internal_ids = self.user_id_converter.convert_external_to_internal(bin_with_external_ids)

        return self.similarity_matrix_slice_getter[achor_user_internal_id, bin_with_internal_ids] 

    def __call__(self, user_ids_bin: Sequence, 
                 anchor_user_id: int) -> Tuple[Sequence, Sequence]:
        user_external_ids_bin = user_ids_bin
        similarities = self.get_similarities(anchor_user_id, user_external_ids_bin)
        return sparate_bin_on_max_distance(user_external_ids_bin, similarities, self.min_elements_in_bin)
