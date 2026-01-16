import torch
import torch.nn as nn

class TwoPartEmbedding(nn.Module):
    def __init__(self, embed_1: nn.Embedding, embed_2: nn.Embedding, 
                 emb_size_must_be_equal: bool = True) -> None:
        super().__init__()

        if emb_size_must_be_equal and not embed_1.embedding_dim == embed_2.embedding_dim:
            raise ValueError(f'embed_1.embedding_dim = {embed_1.embedding_dim} != embed_2.embedding_dim = {embed_2.embedding_dim}')

        self.embed_1 = embed_1
        self.embed_2 = embed_2
        

    def forward(self, indices: torch.Tensor):
        offset = self.embed_1.num_embeddings
        mask_embed_1 = indices < offset
        mask_embed_2 = indices >= offset

        embeddings_1 = self.embed_1(indices[mask_embed_1])

        adjusted_indices_2 = indices[mask_embed_2] - offset
        embeddings_2 = self.embed_2(adjusted_indices_2)

        result = torch.zeros(indices.shape[0], self.embed_1.embedding_dim, device=indices.device)

        result[mask_embed_1] = embeddings_1
        result[mask_embed_2] = embeddings_2

        return result
    
    @property
    def num_embeddings(self):
        return self.embed_1.num_embeddings + self.embed_2.num_embeddings
