import numpy as np
from typing import List, Iterable, Optional, Set


def create_bins_for_user(values: np.ndarray, 
                         n_bins: int, 
                         exclude_id: Optional[int] = None) -> List[Set[int]]:
    if exclude_id is not None:
        mask = np.arange(values.shape[0]) != exclude_id
        indexed_values = np.arange(values.shape[0])[mask]
        values = values[mask]
    else:
        indexed_values = np.arange(values.shape[0])

    min_val, max_val = np.min(values), np.max(values)
    bin_size = (max_val - min_val) / n_bins

    bins = [set() for _ in range(n_bins)]
    
    eps = 1e-6
    bin_indices = np.floor((values - min_val) / (bin_size + eps)).astype(int)

    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    for idx, bin_idx in zip(indexed_values, bin_indices):
        bins[bin_idx].add(idx)
        
    return bins



# This implementation is for functional tests only.
def create_bins_for_user__no_vectorization(values: Iterable[int], 
                         n_bins: int, 
                         exclude_id: Optional[int] = None) -> List[Set[int]]:
    if exclude_id is not None:
        mask = np.arange(values.shape[0]) != exclude_id
        indexed_values = np.arange(values.shape[0])[mask]
        values = values[mask]
    else:
        indexed_values = np.arange(values.shape[0])

    min_val, max_val = np.min(values), np.max(values)
    bin_size = (max_val - min_val) / n_bins

    bins = [set() for _ in range(n_bins)]
    
    eps = 1e-6
    bin_indices = np.floor((values - min_val) / (bin_size + eps)).astype(int)

    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    for idx, bin_idx in zip(indexed_values, bin_indices):
        bins[bin_idx].add(idx)
        
    return bins
