from typing import Tuple

import torch
import pytorch_lightning as pl
from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.data_load.padded_batch import PaddedBatch  # for typing

from ptls_extension_2024_research import TrxEncoder_WithCIEmbeddings
from ptls_extension_2024_research.nn.trx_encoder.client_item_encoder import StaticGNNTrainableClientItemEncoder
from ptls_extension_2024_research.frames.coles_client_id_aware.coles_module__trx_with_ci_embs import CoLESModule_CITrx
from ptls_extension_2024_research.frames.gnn.gnn_module import GnnModule
from ptls_extension_2024_research.graphs.utils import RandEdgeSamplerFull
from ptls_extension_2024_research.lightning_utlis import LogLstEl


def get_ci_embedder_from_seq_encoder(seq_encoder):
    trx_encoder = seq_encoder.trx_encoder
    assert isinstance(trx_encoder, TrxEncoder_WithCIEmbeddings), f"Unexpected trx_encoder type: {type(trx_encoder)}"
    gnns_ci_embedders = [embedder for embedder in trx_encoder.client_item_embeddings if isinstance(embedder, StaticGNNTrainableClientItemEncoder)]
    assert len(gnns_ci_embedders) == 1, f"Unexpected number of GNNClientItemEncoder instances: {len(gnns_ci_embedders)}"
    return gnns_ci_embedders[0]

class ColesGnnModule(pl.LightningModule):
    """
    Arguments:
    ----------
    neg_edge_sampler is an object of an edge sampler calss
    """
    def __init__(self,
                 seq_encoder: SeqEncoderContainer,
                 loss_gamma: float = 0.5,
                 coles_head=None,
                 coles_loss=None,
                 coles_validation_metric=None,
                 neg_items_per_pos = 1,
                 lr_criterion_name = 'BCELoss',
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 neg_edge_sampler=None):
        super().__init__()

        # ptls.training_module.py saves seq_encoder stored inside lightning module.
        # So we have to have this property.
        self.seq_encoder = seq_encoder

        gnn = self.get_gnn_from_seq_encoder(seq_encoder)
        self.coles_module = CoLESModule_CITrx(seq_encoder, coles_head, coles_loss, coles_validation_metric, 
                                           optimizer_partial=None, lr_scheduler_partial=None)
        self.gnn_module = GnnModule(gnn, optimizer_partial=None, lr_scheduler_partial=None, 
                                    neg_edge_sampler=neg_edge_sampler,
                                    neg_items_per_pos = neg_items_per_pos, 
                                    lr_criterion_name = lr_criterion_name)
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial
        self.loss_gamma = loss_gamma
    
    def get_ci_embedder_from_seq_encoder(self, seq_encoder):
        return get_ci_embedder_from_seq_encoder(seq_encoder)

    def get_gnn_from_seq_encoder(self, seq_encoder):
        return self.get_ci_embedder_from_seq_encoder(seq_encoder).gnn_link_predictor

    # def forward(self, x):
    #     pass

    def training_step(self, batch, batch_idx):
        subgraph = self.get_subgraph(batch)
        gnn_loss = self.gnn_module.training_step(subgraph, batch_idx)
        coles_loss = self.coles_module.training_step(batch, batch_idx)
        full_loss = self.loss_gamma * coles_loss + (1-self.loss_gamma) * gnn_loss
        return full_loss
        

    def validation_step(self, batch, batch_idx):
        self.coles_module.validation_step(batch, batch_idx)
        self.gnn_module.validation_step(batch, batch_idx) 

    def on_validation_epoch_end(self):
        self.coles_module.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]


    





class ColesGnnModuleFullGraph(pl.LightningModule):
    """
    A special case of ColesGnnModule where RandEdgeSamplerFull is used
    """
    def __init__(self,
                seq_encoder: SeqEncoderContainer,
                freeze_embeddings_outside_coles_batch: bool,
                include_gnn_users_in_contrastive_loss: bool,
                use_gnn_loss: bool,
                loss_gamma: float = 0.5,
                gnn_loss_alpha: float = 0.5,
                coles_head=None,
                coles_loss=None,
                coles_validation_metric=None,
                rand_edge_sampler_seed = None,
                neg_items_per_pos = 1,
                lp_criterion_name = 'BCELoss',
                optimizer_partial=None,
                lr_scheduler_partial=None) -> None:
        super().__init__()  # calling "GrandParent's" init
        if not use_gnn_loss and loss_gamma != 1:
            print(f"Warning! loss_gamma values = {loss_gamma} will be ignored since use_gnn_loss = False")


        # ptls.training_module.py saves seq_encoder stored inside lightning module.
        # So we have to have this property.
        self.seq_encoder = seq_encoder

        self.coles_module = CoLESModule_CITrx(seq_encoder, coles_head, coles_loss, coles_validation_metric, 
                                           optimizer_partial=None, lr_scheduler_partial=None)
        
        ci_embedder = self.get_ci_embedder_from_seq_encoder(seq_encoder)
        gnn = ci_embedder.gnn_link_predictor
        self.data_adapter = ci_embedder.data_adapter
        self.client_item_g = ci_embedder.data_adapter.client_item_g
        rand_edge_sampler = RandEdgeSamplerFull(
            self.client_item_g.g, rand_edge_sampler_seed)
        
        self.gnn_module = GnnModule(gnn, optimizer_partial=None, lr_scheduler_partial=None, 
                                    neg_edge_sampler=rand_edge_sampler,
                                    gnn_loss_alpha=gnn_loss_alpha,
                                    neg_items_per_pos = neg_items_per_pos, 
                                    lp_criterion_name = lp_criterion_name)
        
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial
        self.loss_gamma = loss_gamma
        self.freeze_embeddings_outside_coles_batch = freeze_embeddings_outside_coles_batch
        self.col_item_ids = seq_encoder.trx_encoder.col_item_ids
        self.include_gnn_users_in_contrastive_loss = include_gnn_users_in_contrastive_loss
        self.linear_for_gnn_users_contrastive_loss = None
        if self.include_gnn_users_in_contrastive_loss:
            coles_h_dim = seq_encoder.seq_encoder.hidden_size
            gnn_h_dim = gnn._output_size
            self.linear_for_gnn_users_contrastive_loss = torch.nn.Linear(gnn_h_dim, coles_h_dim)
        self.use_gnn_loss = use_gnn_loss

        # Is used in on_before_optimizer_step 
        # if `freeze_embeddings_outside_coles_batch` == True
        self.current_seq_len_mask = None


    def get_ci_embedder_from_seq_encoder(self, seq_encoder):
        return get_ci_embedder_from_seq_encoder(seq_encoder)

    def get_gnn_from_seq_encoder(self, seq_encoder):
        return self.get_ci_embedder_from_seq_encoder(seq_encoder).gnn_link_predictor

    # def forward(self, x):
    #     pass

    def coles_batch_to_client_and_item_ids(self, batch: Tuple[PaddedBatch, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, client_ids = batch
        item_ids = x.payload[self.col_item_ids]
        return client_ids, item_ids
    
    def convert_coles_ids_to_graph_ids(self, client_ids, item_ids):
        item_ids = self.data_adapter.item_id2graph_id[item_ids]
        client_ids = self.data_adapter.client_id2graph_id[client_ids]
        return client_ids, item_ids


    def training_step(self, batch, batch_idx):
        coles_client_ids, coles_item_ids = self.coles_batch_to_client_and_item_ids(batch)
        coles_unique_client_ids = torch.unique(coles_client_ids)
        self.current_graph_client_ids, self.current_graph_item_ids = self.convert_coles_ids_to_graph_ids(coles_unique_client_ids, coles_item_ids)
        
        if self.freeze_embeddings_outside_coles_batch:
            # Will be used in  `self.on_before_optimizer_step`.
            self.current_seq_len_mask = batch[0].seq_len_mask


        ###### Graph training_step code
        gnn_log_list = []
        subgraph = self.client_item_g.create_subgraph(self.current_graph_client_ids, self.current_graph_item_ids)
        subgraph_node_embeddings = self.gnn_module.gnn_link_predictor(subgraph)
        
        if self.use_gnn_loss:
            gnn_loss, gnn_auc = self.gnn_module.calc_loss(subgraph, subgraph_node_embeddings)
            gnn_log_list.append(LogLstEl('loss', gnn_loss))
            gnn_log_list.append(LogLstEl('lp_auc', gnn_auc))


        ####### Coles training_step code
        coles_log_list = []
        coles_embeddings, y = self.coles_module.shared_step(*batch)


        if self.include_gnn_users_in_contrastive_loss:
            # Достать эмбеддинги пользователей, которые в эмбеддинг матрице графа имеют id self.current_graph_client_ids.

            # Если будет не полный граф, а подграфы, нужно будет использовать
            # `subgraph_id_to_graph_id = subgraph.ndata['_ID']`
            graph_client_embeddings_from_coles_batch = subgraph_node_embeddings[self.current_graph_client_ids]
            scales_graph_embs = self.linear_for_gnn_users_contrastive_loss(graph_client_embeddings_from_coles_batch)
            coles_embeddings = torch.cat([coles_embeddings, scales_graph_embs])
            y = torch.cat([y, coles_unique_client_ids])


        coles_loss = self.coles_module._loss(coles_embeddings, y)

        coles_log_list.append(LogLstEl('loss', coles_loss))
        coles_log_list.append(self.coles_module.get_seq_len_log_lst_el(batch))


        full_log_list = ([el.alter_name(lambda x: f"coles/{x}") for el in coles_log_list] + 
                    [el.alter_name(lambda x: f"gnn/{x}") for el in gnn_log_list])

        if self.use_gnn_loss:
            full_loss = self.loss_gamma * coles_loss + (1-self.loss_gamma) * gnn_loss
            full_log_list.append(LogLstEl('full_loss', full_loss))
        else:
            full_loss = coles_loss

        # All logging
        for el in full_log_list:
            self.log(el.name, el.value, *el.args, **el.kwargs)
        return full_loss
        
    
    def on_before_optimizer_step(self, optimizer) -> None:
        if not self.freeze_embeddings_outside_coles_batch:
            return
        
        is_present_mask = self.current_seq_len_mask.reshape(-1).to(torch.bool)
        item_ids = self.current_graph_item_ids.reshape(-1)[is_present_mask]

        node_feats = self.gnn_module.gnn_link_predictor.node_feats
        freeze_mask = torch.ones_like(node_feats.weight, dtype=torch.bool)
        freeze_mask[item_ids] = False
        freeze_mask[self.current_graph_client_ids.reshape(-1)] = False
        node_feats.weight.grad[freeze_mask] = 0

    def validation_step(self, batch, batch_idx):
        # Не стал доставать графовые client_ids и item_ids из батча.
        # Нет мысла в данном сценарии....
        dummy_client_ids = None
        dummy_item_ids = None

        subgraph = self.client_item_g.create_subgraph(dummy_client_ids, dummy_item_ids)
        self.coles_module.validation_step(batch, batch_idx)
        gnn_loss, gnn_auc = self.gnn_module.validation_step(subgraph, batch_idx)
        self.log('gnn/valid/gnn_loss', gnn_loss)
        self.log('gnn/valid/lp_auc', gnn_auc)

    def _on_validation_epoch_end__coles(self):
        self.log(f'coles/valid/{self.coles_module.metric_name}', self.coles_module._validation_metric.compute(), prog_bar=True)
        self.coles_module._validation_metric.reset()

    def on_validation_epoch_end(self):
        self._on_validation_epoch_end__coles()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                'scheduler': scheduler,
                'monitor': self.metric_name,
            }
        return [optimizer], [scheduler]




"""
* shared_step - общий шаг для обучения и валидации: принимает фичи и user_ids, возвращает user_embeddings и user_ids
* forward - принимает фичи, возвращает user_embeddings
* training_step - шаг обучения: принимает батч и индекс батча, возвращает лосс
* validation_step - шаг валидации: принимает батч и индекс батча, вычисляет мтерики, ничего не возвращает
* on_validation_epoch_end - логирует метрики, вычисленные на валидации
* configure_optimizers - конфигурирует оптимизаторы и lr_scheduler'ы
"""






# class ABSModule(pl.LightningModule):
#     @property
#     def metric_name(self):
#         raise NotImplementedError()

#     @property
#     def is_requires_reduced_sequence(self):
#         raise NotImplementedError()

#     def shared_step(self, x, y):
#         """

#         Args:
#             x:
#             y:

#         Returns: y_h, y

#         """
#         raise NotImplementedError()

#     def __init__(self, validation_metric=None,
#                        seq_encoder=None,
#                        loss=None,
#                        optimizer_partial=None,
#                        lr_scheduler_partial=None):
#         """
#         Parameters
#         ----------
#         params : dict
#             params for creating an encoder
#         seq_encoder : torch.nn.Module
#             sequence encoder, if not provided, will be constructed from params
#         """
#         super().__init__()
#         # self.save_hyperparameters()

#         self._loss = loss
#         self._seq_encoder = seq_encoder
#         self._seq_encoder.is_reduce_sequence = self.is_requires_reduced_sequence
#         self._validation_metric = validation_metric

#         self._optimizer_partial = optimizer_partial
#         self._lr_scheduler_partial = lr_scheduler_partial

#     @property
#     def seq_encoder(self):
#         return self._seq_encoder

#     def forward(self, x):
#         return self._seq_encoder(x)

#     def training_step(self, batch, _):
#         y_h, y = self.shared_step(*batch)
#         loss = self._loss(y_h, y)
#         self.log('loss', loss)
#         if type(batch) is tuple:
#             x, y = batch
#             if isinstance(x, PaddedBatch):
#                 self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
#         else:
#             # this code should not be reached
#             self.log('seq_len', -1, prog_bar=True)
#             raise AssertionError('batch is not a tuple')
#         return loss

#     def validation_step(self, batch, _):
#         y_h, y = self.shared_step(*batch)
#         self._validation_metric(y_h, y)

#     def on_validation_epoch_end(self):
#         self.log(f'valid/{self.metric_name}', self._validation_metric.compute(), prog_bar=True)
#         self._validation_metric.reset()

#     def configure_optimizers(self):
#         optimizer = self._optimizer_partial(self.parameters())
#         scheduler = self._lr_scheduler_partial(optimizer)
        
#         if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler = {
#                 'scheduler': scheduler,
#                 'monitor': self.metric_name,
#             }
#         return [optimizer], [scheduler]
