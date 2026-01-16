import torch
import torch.nn as nn

from ptls_extension_2024_research.frames.gnn.gnn_module import ColesBatchToSubgraphConverter, GnnLinkPredictor


class BaseClientItemEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        raise NotImplementedError()

    @property
    def output_size(self) -> int:
        raise NotImplementedError()


class DummyGNNClientItemEncoder(BaseClientItemEncoder):
    def __init__(self, output_size = 10):
        super().__init__()
        self.__output_size = output_size

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        batch_size, seq_len = item_ids.size()
        return torch.zeros(batch_size, seq_len, self.__output_size, device=item_ids.device)

    @property
    def output_size(self):
        return self.__output_size



class StaticGNNTrainableClientItemEncoder(BaseClientItemEncoder):
    def __init__(self,
                 data_adapter: ColesBatchToSubgraphConverter,
                 gnn_link_predictor: GnnLinkPredictor,) -> None:
        super().__init__()
        self.gnn_link_predictor = gnn_link_predictor
        self.data_adapter = data_adapter

    def forward(self, client_ids: torch.Tensor, item_ids: torch.Tensor):
        """
        client_ids: torch.Tensor, shape: (batch_size,)
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        data_adapter_result = self.data_adapter(client_ids, item_ids)
        subgraph, subgraph_item_ids = \
            (data_adapter_result['subgraph'],
             data_adapter_result['subgraph_item_ids'])
        subgraph_node_embeddings = self.gnn_link_predictor(subgraph)
        item_embeddings = subgraph_node_embeddings[subgraph_item_ids]
        return item_embeddings
    
    @property
    def output_size(self):
        return self.gnn_link_predictor._output_size
    