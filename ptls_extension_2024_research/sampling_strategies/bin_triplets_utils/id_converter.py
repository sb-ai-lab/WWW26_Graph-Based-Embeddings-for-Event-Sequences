from typing import Dict

import numpy as np
import torch

class IdConverterBase:
    def convert_external_to_internal(self, external_ids):
        raise NotImplementedError

    def convert_internal_to_external(self, external_ids):
        raise NotImplementedError


class IdConverter_Dict(IdConverterBase):
    def __init__(self, internal_id_to_external_id: Dict[int, int]) -> None:
        self.internal_id_to_external_id = internal_id_to_external_id
        self.external_id_to_internal_id = {v: k for k,v in self.internal_id_to_external_id.items()}
    
    def convert_external_to_internal(self, external_ids):
        if isinstance(external_ids, int) or isinstance(external_ids, np.integer):
            return self.external_id_to_internal_id[external_ids]
        return [self.external_id_to_internal_id[external_id] for external_id in external_ids]
    
    def convert_internal_to_external(self, internal_ids):
        if isinstance(internal_ids, int) or isinstance(internal_ids, np.integer):
            return self.internal_id_to_external_id[internal_ids]
        return [self.internal_id_to_external_id[internal_id] for internal_id in internal_ids]
    
    @classmethod
    def from_external_id_to_internal_id(cls, external_id_to_internal_id: Dict[int, int]) -> None:
        internal_id_to_external_id = {v: k for k,v in external_id_to_internal_id.items()}
        return cls(internal_id_to_external_id)


class IdConverter_Array:
    def __init__(self, internal_id_to_external_id: Dict[int, int]) -> None:
        internal_ids, external_ids = zip(*internal_id_to_external_id.items())
        max_internal_id = max(internal_ids)
        invalid_internal_id = max_internal_id + 1
        max_external_id = max(external_ids)
        invalid_external_id = max_external_id + 1
        
        self.external_id_to_internal_id = np.full([max_external_id], invalid_internal_id)
        self.external_id_to_internal_id[external_ids] = np.array(internal_ids)

        self.internal_id_to_external_id = np.full([max_internal_id], invalid_external_id)
        self.internal_id_to_external_id[internal_ids] = np.array(external_ids)
    

    def convert_external_to_internal(self, external_ids):
        return self.external_id_to_internal_id[external_ids]
    

    def convert_internal_to_external(self, internal_ids):
        return self.internal_id_to_external_id[internal_ids]



class IdConverter_Tensor:
    def __init__(self, internal_id_to_external_id: Dict[int, int], device_name: str) -> None:
        device = torch.device(device_name)

        internal_ids, external_ids = zip(*internal_id_to_external_id.items())
        max_internal_id = max(internal_ids)
        invalid_internal_id = max_internal_id + 1
        max_external_id = max(external_ids)
        invalid_external_id = max_external_id + 1
        
        self.external_id_to_internal_id = torch.full([max_external_id], invalid_internal_id, dtype=torch.int32, device=device)
        self.external_id_to_internal_id[external_ids] = torch.tensor(internal_ids)

        self.internal_id_to_external_id = torch.full([max_internal_id], invalid_external_id, dtype=torch.int32, device=device)
        self.internal_id_to_external_id[internal_ids] = torch.tensor(external_ids)


    def convert_external_to_internal(self, external_ids):
        return self.external_id_to_internal_id[external_ids]
    

    def convert_internal_to_external(self, internal_ids):
        return self.internal_id_to_external_id[internal_ids]
    