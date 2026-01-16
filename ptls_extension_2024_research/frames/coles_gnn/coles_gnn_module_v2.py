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
from ptls_extension_2024_research.frames.gnn.gnn_module_v2 import GnnModule, GnnEmbedder
from ptls_extension_2024_research.graphs.utils import RandEdgeSamplerFull
from ptls_extension_2024_research.lightning_utlis import LogLstEl
from ptls_extension_2024_research.graphs.graph import ClientItemGraph, ClientItemGraphFull
from ptls_extension_2024_research.nn.trx_encoder.encoders import EmbeddingEncoder



def get_ci_embedder_from_seq_encoder(seq_encoder):
    trx_encoder = seq_encoder.trx_encoder
    assert isinstance(trx_encoder, TrxEncoder_WithCIEmbeddings), f"Unexpected trx_encoder type: {type(trx_encoder)}"
    gnns_ci_embedders = [embedder for embedder in trx_encoder.client_item_embeddings if isinstance(embedder, StaticGNNTrainableClientItemEncoder)]
    assert len(gnns_ci_embedders) == 1, f"Unexpected number of GNNClientItemEncoder instances: {len(gnns_ci_embedders)}"
    return gnns_ci_embedders[0]



class IdConverter:
    def __init__(self, 
                 item_id2graph_id: torch.Tensor, 
                 client_id2graph_id: torch.Tensor, 
                 device_name: str) -> None:
        super().__init__()
        self.device = torch.device(device_name)
        self.item_id2graph_id = item_id2graph_id.to(self.device)
        self.client_id2graph_id = client_id2graph_id.to(self.device)
     
    def convert_ptls_item_ids_to_graph_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        graph_item_ids = self.item_id2graph_id[item_ids]
        return graph_item_ids
     
    def convert_ptls_client_ids_to_graph_ids(self, client_ids: torch.Tensor) -> torch.Tensor:
        graph_client_ids = self.client_id2graph_id[client_ids]
        return graph_client_ids

    def invert_tensor_based_map(self, torch_map: torch.Tensor, invalid_index_val: int = -1) -> torch.Tensor:
        inverted_map = torch.full((max(torch_map) + 1,), invalid_index_val)
        inverted_map[torch_map] = torch.arange(torch_map.shape[0])
        return inverted_map

    def covnvert_ptls_item_ids_to_subgraph_ids(self, 
                                               subgraph_ids_to_graph_ids: torch.Tensor, 
                                               ptls_item_ids: torch.Tensor) -> torch.Tensor:
        graph_ids_to_subgraph_ids = self.invert_tensor_based_map(subgraph_ids_to_graph_ids)
        item_graph_ids = self.item_id2graph_id[ptls_item_ids]
        item_subgraph_ids = graph_ids_to_subgraph_ids[item_graph_ids]
        return item_subgraph_ids
   



class ColesGnnModuleFullGraph(pl.LightningModule):
    """
    A special case of ColesGnnModule where RandEdgeSamplerFull is used
    """
    def __init__(self,
                seq_encoder: SeqEncoderContainer,
                gnn_embedder: GnnEmbedder,
                id_converter: IdConverter,
                client_item_g: ClientItemGraphFull,
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
                link_predictor_name='MLP',
                link_predictor_add_sigmoid: bool=True, 
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
        self.id_converter = id_converter
        self.client_item_g = client_item_g
        rand_edge_sampler = RandEdgeSamplerFull(
            self.client_item_g.g, rand_edge_sampler_seed)
        
        self.gnn_module = GnnModule(gnn_embedder, optimizer_partial=None, lr_scheduler_partial=None, 
                                    neg_edge_sampler=rand_edge_sampler,
                                    gnn_loss_alpha=gnn_loss_alpha,
                                    neg_items_per_pos = neg_items_per_pos, 
                                    lp_criterion_name = lp_criterion_name,
                                    link_predictor_name=link_predictor_name,
                                    link_predictor_add_sigmoid=link_predictor_add_sigmoid)
        
        self._assert_n_embeddings_in_gnn_feats_is_correct()

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial
        self.loss_gamma = loss_gamma
        # self.freeze_embeddings_outside_coles_batch = freeze_embeddings_outside_coles_batch
        self.col_item_ids = seq_encoder.trx_encoder.col_item_ids
        self.include_gnn_users_in_contrastive_loss = include_gnn_users_in_contrastive_loss
        self._init_linear_for_gnn_users_contrastive_loss(gnn_embedder._output_size)
        self.custom_item_id_embedder = self._get_custom_item_id_embedder()
        self.use_gnn_loss = use_gnn_loss


    def _assert_n_embeddings_in_gnn_feats_is_correct(self) -> None:
        n_client_feats_gnn = self.gnn_module.gnn_embedder.client_embeddings.num_embeddings
        n_item_feats_gnn = self.gnn_module.gnn_embedder.item_embeddings.num_embeddings
        assert n_client_feats_gnn + n_item_feats_gnn == self.client_item_g.g.num_nodes()

    def _init_linear_for_gnn_users_contrastive_loss(self, gnn_embedder_output_size):
        self.linear_for_gnn_users_contrastive_loss = None
        if self.include_gnn_users_in_contrastive_loss:
            coles_h_dim = self.seq_encoder.seq_encoder.hidden_size
            gnn_h_dim = gnn_embedder_output_size
            self.linear_for_gnn_users_contrastive_loss = torch.nn.Linear(gnn_h_dim, coles_h_dim)

    def get_ci_embedder_from_seq_encoder(self, seq_encoder):
        return get_ci_embedder_from_seq_encoder(seq_encoder)

    # def forward(self, x):
    #     pass

    def coles_batch_to_client_and_item_ids(self, batch: Tuple[PaddedBatch, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, client_ids = batch
        item_ids = x.payload[self.col_item_ids]
        return client_ids, item_ids
    
    def _get_custom_item_id_embedder(self) -> EmbeddingEncoder:
        custom_item_id_embedder = self.coles_module.seq_encoder.trx_encoder.custom_embeddings[self.col_item_ids]
        assert isinstance(custom_item_id_embedder, EmbeddingEncoder), \
            f'Expected custom_item_id_embedder to be an instance of EmbeddingEncoder, ' \
                f'got `{type(custom_item_id_embedder)}` type'
        return custom_item_id_embedder


    def training_step(self, batch, batch_idx):
        client_ptls_ids, item_ptls_ids = self.coles_batch_to_client_and_item_ids(batch)
        unique_client_ptls_ids, unique_item_ptls_ids = client_ptls_ids.unique(), item_ptls_ids.flatten().unique()
        current_graph_unique_client_ids = self.id_converter.convert_ptls_client_ids_to_graph_ids(unique_client_ptls_ids)
        current_graph_unique_item_ids = self.id_converter.convert_ptls_item_ids_to_graph_ids(unique_item_ptls_ids)

        ###### Graph training_step code
        gnn_log_list = []
        subgraph = self.client_item_g.create_subgraph(current_graph_unique_client_ids, current_graph_unique_item_ids)
        subgraph_node_embeddings = self.gnn_module.gnn_embedder(subgraph)
        
        if self.use_gnn_loss:
            gnn_loss, gnn_auc = self.gnn_module.calc_loss(subgraph, subgraph_node_embeddings)
            gnn_log_list.append(LogLstEl('loss', gnn_loss))
            gnn_log_list.append(LogLstEl('lp_auc', gnn_auc))


        unique_item_subgraph_ids = self.id_converter.covnvert_ptls_item_ids_to_subgraph_ids(
            subgraph_ids_to_graph_ids=subgraph.ndata['_ID'], ptls_item_ids=unique_item_ptls_ids)        
        graph_item_embeddings = subgraph_node_embeddings[unique_item_subgraph_ids]
        self.custom_item_id_embedder.update_embeddings(unique_item_ptls_ids, graph_item_embeddings)



        ####### Coles training_step code
        coles_log_list = []
        coles_embeddings, y = self.coles_module.shared_step(*batch)


        # Detach "embedding storage" from computational graph
        # after a copy of embedings_of_interest is extracted and used.
        self.custom_item_id_embedder.detach()


        if self.include_gnn_users_in_contrastive_loss:
            # Достать эмбеддинги пользователей, которые в эмбеддинг матрице графа имеют id self.current_graph_client_ids.

            # Если будет не полный граф, а подграфы, нужно будет current_subgraph_client_ids
            # `subgraph_id_to_graph_id = subgraph.ndata['_ID']`
            graph_client_embeddings_from_coles_batch = subgraph_node_embeddings[current_graph_unique_client_ids]
            scales_graph_embs = self.linear_for_gnn_users_contrastive_loss(graph_client_embeddings_from_coles_batch)
            coles_embeddings = torch.cat([coles_embeddings, scales_graph_embs])
            y = torch.cat([y, unique_client_ptls_ids])


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
    
    

    def validation_step(self, batch, batch_idx):
        # Не стал доставать графовые client_ids и item_ids из батча.
        # Нет мысла в данном сценарии....
        dummy_client_ids = None
        dummy_item_ids = None

        subgraph = self.client_item_g.create_subgraph(dummy_client_ids, dummy_item_ids)
        self.coles_module.validation_step(batch, batch_idx)
        # gnn_loss, gnn_auc = self.gnn_module.validation_step(subgraph, batch_idx)
        # self.log('gnn/valid/gnn_loss', gnn_loss)
        # self.log('gnn/valid/lp_auc', gnn_auc)

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

