from typing import Optional
import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from ptls.nn.trx_encoder.batch_norm import RBatchNorm, RBatchNormWithLens

from ptls_extension_2024_research.nn.trx_encoder.encoders import PretrainedGraphItemEmbedder
from ptls_extension_2024_research.frames.coles_gnn.coles_gnn_module import ColesGnnModuleFullGraph

logger = logging.getLogger(__name__)


def get_gnn_embeddings(coles_gnn_module) -> torch.Tensor:
    full_graph = coles_gnn_module.client_item_g.g
    return coles_gnn_module.gnn_module.gnn_link_predictor(full_graph)


def get_gnn_pretrained_embed_layer(coles_gnn_module, device: Optional[torch.device] = None, freeze: bool = False) -> torch.nn.Embedding:
    gnn_embeddings = get_gnn_embeddings(coles_gnn_module)

    # These parameters can be anything since we just need the weights
    item_id2graph_id = coles_gnn_module.data_adapter.item_id2graph_id

    if device is None:
        device = item_id2graph_id.device

    return PretrainedGraphItemEmbedder(gnn_embeddings, item_id2graph_id, device, freeze)


def set_custom_embedding_batch_norm(coles, conf) -> None:
    if coles.seq_encoder.trx_encoder.custom_embedding_batch_norm == None:
        return
    assert 'custom_embedding_batch_norm_action' in conf
    if conf.custom_embedding_batch_norm_action == 'delete':
        coles.seq_encoder.trx_encoder.custom_embedding_batch_norm = None
    if conf.custom_embedding_batch_norm_action == 'replace':
        emb_size = coles.seq_encoder.trx_encoder.custom_embedding_size
        current_batch_norm = coles.seq_encoder.trx_encoder.custom_embedding_batch_norm
        batch_norm_cls = type(current_batch_norm)
        assert batch_norm_cls in [RBatchNorm, RBatchNormWithLens]
        coles.seq_encoder.trx_encoder.custom_embedding_batch_norm = batch_norm_cls(emb_size)

def set_coles_pretrained_embedding(coles, pretrained_embed_layer, item_col, conf):
    coles.seq_encoder.trx_encoder.custom_embeddings[item_col] = pretrained_embed_layer
    set_custom_embedding_batch_norm(coles, conf)


def coles_gnn__to__coles_with_pretrained_embed_layer(coles_gnn_module: ColesGnnModuleFullGraph, 
                                                     conf):
    coles = coles_gnn_module.coles_module
    pretrained_embed_layer = get_gnn_pretrained_embed_layer(coles_gnn_module, conf.device, conf.freeze_gnn_pretrained_layer)
    # item_col can be inferred from trx_encoder
    item_id_col = coles.seq_encoder.trx_encoder.col_item_ids
    set_coles_pretrained_embedding(coles, pretrained_embed_layer, item_id_col, conf)
    # TODO: Обойти все элементы seq_encoder.trx_encoder.client_item_embeddings
    # и если ci_embedder содержит ту же gnn, что в gnn_module, удалить элемент из списка. 
    coles.seq_encoder.trx_encoder.client_item_embeddings = torch.nn.ModuleList([])
    return coles


def prepare_optimizer_state(ckpt, conf) -> None:
    if conf['optimizer_state_action'] == 'delete':
        del ckpt['optimizer_states']
    else:
        raise NotImplementedError


@hydra.main(version_base='1.2', config_path=None, config_name=None)
def main(conf: DictConfig):
    ckpt = torch.load(conf.ckpt_path, map_location=conf.device)
    coles_gnn_module = hydra.utils.instantiate(conf.pl_module)
    coles_gnn_module.load_state_dict(ckpt['state_dict'])

    coles = coles_gnn__to__coles_with_pretrained_embed_layer(coles_gnn_module, conf)

    ckpt['state_dict'] = coles.state_dict()

    prepare_optimizer_state(ckpt, conf)

    os.makedirs(os.path.dirname(conf['updated_ckpt_path']), exist_ok = True) 
    torch.save(ckpt, conf['updated_ckpt_path'])


if __name__ == '__main__':
    main()
