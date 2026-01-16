from functools import wraps
from typing import List, Union
import hashlib

import numpy as np


def _hash_input_v2(input_data: Union[List, np.ndarray]) -> str:
    if isinstance(input_data, np.ndarray):
        input_data_bytes = input_data.tobytes()
    else:
        input_data_bytes = str(input_data).encode('utf-8')
    return hashlib.md5(input_data_bytes).hexdigest()

def _hash_input__convertion_to_tuple(input_data: Union[List, np.ndarray]) -> str:
    return hash(tuple(input_data))

HASH_INPUT_DEFAULT_FUNC = _hash_input__convertion_to_tuple



def np_like_cache_of_size_one(func):
    cache = {"key": None, "value": None}  # Store only one key-value pair
    
    @wraps(func)
    def wrapper(input_data: Union[List, np.ndarray]):
        cache_key = HASH_INPUT_DEFAULT_FUNC(input_data)
        
        if cache["key"] == cache_key:
            return cache["value"]
        
        result = func(input_data)
        cache["key"] = cache_key
        cache["value"] = result
        return result
    
    return wrapper
