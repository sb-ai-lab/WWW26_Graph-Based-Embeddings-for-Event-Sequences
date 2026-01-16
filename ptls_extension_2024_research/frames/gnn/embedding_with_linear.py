import torch
import torch.nn as nn

class EmbeddingWithLinear(nn.Module):
    """
    This is embedding+linear class WITH EMBEDDING INTERFACE
    meaning it has num_embeddings, embedding_dim properties (maybe more).

    It is to be used in gnn_embedder 
    """
    def __init__(self, embedding_layer: nn.Embedding, out_features: int) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        linear_in_features = embedding_layer.embedding_dim
        self.linear = nn.Linear(linear_in_features, out_features)
        
        self.embedding_dim = out_features
        self.num_embeddings = embedding_layer.num_embeddings

    def forward(self, idxs):
        embs = self.embedding_layer(idxs)
        return self.linear(embs)
