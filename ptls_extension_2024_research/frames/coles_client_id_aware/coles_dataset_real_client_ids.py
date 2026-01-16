from typing import List, Dict, Tuple, Optional
from functools import reduce
from operator import iadd

import torch
import numpy as np  # for typing

from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit
from ptls.data_load.padded_batch import PaddedBatch  # for typing


SplitsType = List[Dict[str, np.ndarray]]


class ColesDataset(FeatureDict, torch.utils.data.Dataset):
    """
    This is different from ptls.frames.coles.ColesDataset in two aspects:
    * Collate_fn returns REAL client_ids instead of 
      just different integers for different clients retirieved via enumerate.
    * An i-th dataset element contains not only n dicts representing 
      splits of sequential features, but also an id. This is required
      to get real ids in collate_fn. Even if we don't take client_ids 
      from the table but consider index in a dataset as an id, we can't get
      theese indexes in `collate_fn` without making them a part of a dataset element.
      
    Dataset for `ptls_extension_2024_research.frames.coles_client_aware.CoLESModule_CITrx`

    Parameters
    ----------
    data:
        source data with feature dicts
    splitter: 
        object from from `ptls.frames.coles.split_strategy`.
        Used to split original sequence into subsequences which are samples from one client.
    col_time:
        column name with event_time
    col_client_id:
        column name with encoded client_id
    """

    def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_client_id: str,
                 col_time: str = 'event_time',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)  # required for mixin class
        self.data = data
        self.splitter = splitter
        self.col_time = col_time
        self.col_client_id = col_client_id

    def __len__(self):
        return len(self.data)
    
    def _get_client_id_from_feature_arrays(self, feature_arrays: dict) -> int:
        client_id = feature_arrays[self.col_client_id]
        if type(client_id) is int:
            return client_id
        if type(client_id) is np.int32:
            return int(client_id)
        # if type(client_id) is str:
        #     if not client_id.isdigit():
        #         raise ValueError("client_id is a string and doesn't represent an integer")
        #     client_id = int(client_id)
        #     return client_id
        raise ValueError(f"client_id is of an unexpected type `{type(client_id)}`. Client id value: {client_id}")
    
    def _get_client_id(self, feature_arrays: dict) -> int:
        return self._get_client_id_from_feature_arrays(feature_arrays)

    def __getitem__(self, idx) -> Tuple[SplitsType, int]:
        """
        Returns a list of n feature dicts (only sequential features)
        sampled from the client with index `idx`.
        """
        feature_arrays = self.data[idx]
        client_id = self._get_client_id(feature_arrays)
        return self.get_splits(feature_arrays), client_id
    
    def __iter__(self):
        for feature_arrays in enumerate(self.data):
            client_id = self._get_client_id(feature_arrays)
            yield self.get_splits(feature_arrays), client_id

    def get_splits(self, feature_arrays):
        """
        Returns a list of n feature dicts (only sequential features). 
        Each dict is sampled from the original `feature_arrays` thus
        we get n samples from one client.
        """
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} for ix in indexes]
        # return [{k: v[ix] if self.is_seq_feature(k, v) else v for k, v in feature_arrays.items() } for ix in indexes]


    @staticmethod
    def collate_fn(batch: List[Tuple[SplitsType, int]]
                   ) -> Tuple[PaddedBatch, torch.Tensor]:
        batch_seqs, client_ids = zip(*batch)

        # ! Myabe it would be better to retrieve split_count via n*
        split_count = len(batch_seqs[0])
        
        # Repeat each id `split_count` times to match batch_seqs.
        client_ids = [client_id for client_id in client_ids for _ in range(split_count)]
        
        # Flatten `List[List[Dict[str, np.ndarray]]]` to `List[Dict[str, np.ndarray]]`.
        batch_seqs = reduce(iadd, batch_seqs)
        padded_batch_seqs = collate_feature_dict(batch_seqs)
        return padded_batch_seqs, torch.LongTensor(client_ids)


class ColesIterableDataset(ColesDataset, torch.utils.data.IterableDataset):
    pass
