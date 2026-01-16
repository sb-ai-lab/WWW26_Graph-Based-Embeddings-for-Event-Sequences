from typing import List, Tuple, Dict
from functools import reduce
from operator import iadd
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.padded_batch import PaddedBatch 
import torch
import numpy as np  # For type hinting.

FeatureDictType = Dict[str, np.ndarray]


class DatasetWithClientId(torch.utils.data.Dataset):
    def __init__(self, 
                 data: torch.utils.data.Dataset,
                 col_client_id: str) -> None:
        print(type(self))
        self.data = data
        self.col_client_id = col_client_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch: List[FeatureDictType]
                   ) -> Tuple[PaddedBatch, torch.Tensor]:
        padded_batch = collate_feature_dict(batch)
        return padded_batch, padded_batch.payload[self.col_client_id]
    

class DatasetWithClientIdIterable(DatasetWithClientId, torch.utils.data.IterableDataset):
    def __iter__(self):
        for data in self.data:
            yield data
    

# class DatasetWithDummyIdTensor(torch.utils.data.Dataset):
#     def __init__(self, data: torch.utils.data.Dataset):
#         self.data = data
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]

#     @staticmethod
#     def collate_fn(batch: List[FeatureDictType]
#                    ) -> Tuple[PaddedBatch, torch.Tensor]:
#         padded_batch = collate_feature_dict(batch)
#         print(padded_batch.payload)
#         return padded_batch, torch.zeros(len(batch))


# class DatasetWithDummyIdTensorIterable(DatasetWithDummyIdTensor, torch.utils.data.IterableDataset):
#     def __iter__(self):
#         for data in self.data:
#             yield data
