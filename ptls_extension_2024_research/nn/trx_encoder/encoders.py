import torch
import torch.nn as nn
from ptls.nn.trx_encoder.encoders import BaseEncoder


class PretrainedGraphItemEmbedder(BaseEncoder):
    def __init__(self, embeddings: torch.Tensor, item_id2graph_id: torch.Tensor,
                 device: torch.device, freeze: bool,) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze).to(device)
        # logger.info(f"{self.embedding_layer.weight.requires_grad = }")
        self.item_id2graph_id = item_id2graph_id.to(device)
        n_nodes, emb_size = embeddings.shape
        self.__output_size = emb_size
    
    def forward(self, item_ids: torch.Tensor):
        """
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        graph_item_ids = self.item_id2graph_id[item_ids]
        return self.embedding_layer(graph_item_ids)
        
    # def unfreeze(self) -> None:
    #     if self.embedding_layer.weight.requires_grad:
    #         print("Warning: trying to unfreeze PretrainedGraphItemEmbedder that is already unfrozen")
    #     self.embedding_layer.requires_grad_(True)

    # def freeze(self) -> None:
    #     if not self.embedding_layer.weight.requires_grad:
    #         print("Warning: trying to freeze PretrainedGraphItemEmbedder that is already frozen")
    #     self.embedding_layer.requires_grad_(False)

    @property
    def output_size(self):
        return self.__output_size
    


class PretrainedEmbeddings(BaseEncoder):
    def __init__(self, embeddings: torch.Tensor, ptls_item_id_to_pretrained_embedding_id: torch.Tensor,
                 device: torch.device, freeze: bool,) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze).to(device)
        # logger.info(f"{self.embedding_layer.weight.requires_grad = }")
        self.ptls_item_id_to_pretrained_embedding_id = ptls_item_id_to_pretrained_embedding_id.to(device)
        n_embeddings, emb_size = embeddings.shape
        self.__output_size = emb_size
    
    def forward(self, item_ids: torch.Tensor):
        """
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        pretrained_embedding_id = self.ptls_item_id_to_pretrained_embedding_id[item_ids]
        return self.embedding_layer(pretrained_embedding_id)
        
    # def unfreeze(self) -> None:
    #     if self.embedding_layer.weight.requires_grad:
    #         print("Warning: trying to unfreeze PretrainedGraphItemEmbedder that is already unfrozen")
    #     self.embedding_layer.requires_grad_(True)

    # def freeze(self) -> None:
    #     if not self.embedding_layer.weight.requires_grad:
    #         print("Warning: trying to freeze PretrainedGraphItemEmbedder that is already frozen")
    #     self.embedding_layer.requires_grad_(False)

    @property
    def output_size(self):
        return self.__output_size



class EmbeddingEncoder(BaseEncoder):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.embeddings_tensor = torch.zeros((num_embeddings, embedding_dim), device=device)
        self.__output_size = embedding_dim
    
    def forward(self, item_ids: torch.Tensor):
        """
        item_ids: torch.Tensor, shape: (batch_size, seq_len)
        """
        return self.embeddings_tensor[item_ids]
        
    def update_embeddings(self, ids: torch.Tensor, embeddings: torch.Tensor) -> None:
        self.embeddings_tensor[ids] = embeddings

    def detach(self) -> None:
        self.embeddings_tensor = self.embeddings_tensor.detach()

    @property
    def output_size(self):
        return self.__output_size