from typing import List

import torch
import pytorch_lightning as pl

class LayersGetter:
    """
    Given pl_module returns a certain list of it's layers. 
    Can be used in .model_modifiers.RequiresGradModifier 
    to select layers to freeze.
    """
    def __call__(self, pl_module: pl.LightningModule) -> List[torch.nn.Module]:
        raise NotImplementedError

class CustomEmbedderGetter(LayersGetter):
    def __init__(self, custom_embedder_key: str):
        self.custom_embedder_key = custom_embedder_key
    
    def __call__(self, pl_module: pl.LightningModule) -> List[torch.nn.Module]:
        return [pl_module.seq_encoder.trx_encoder.custom_embeddings[self.custom_embedder_key]]
