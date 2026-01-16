from typing import List

import torch
import pytorch_lightning as pl

from .layers_getters import LayersGetter
from ..frames.coles_gnn.coles_gnn_module import ColesGnnModule, ColesGnnModuleFullGraph
from ..nn.trx_encoder.trx_encoder_with_client_item_embeddings import TrxEncoder_WithCIEmbeddings


def set_requires_grad_for_layers(layers: List[torch.nn.Module], requires_grad_value: bool) -> None:
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = requires_grad_value


class ModelModifier:
    """
    Given a pl_module modifies it. 
    Can be used to alter pl_modules behavior after n epoches.
    """
    def __call__(self, pl_module: pl.LightningModule) -> None:
        raise NotImplementedError
        

class RequiresGradModifier(ModelModifier):
    def __init__(self, layers_getter: LayersGetter, new_requires_grad_value: bool):
        self._layers_getter = layers_getter
        self._new_requires_grad_value = new_requires_grad_value
    
    def __call__(self, pl_module: pl.LightningModule) -> None:
        set_requires_grad_for_layers(self._layers_getter(pl_module), self._new_requires_grad_value)
