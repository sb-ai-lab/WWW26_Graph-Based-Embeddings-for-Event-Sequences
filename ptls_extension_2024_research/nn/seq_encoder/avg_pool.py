import torch
from ptls.data_load.padded_batch import PaddedBatch


class GlobalAvgPool(torch.nn.Module):
    def forward(self, x_pb: PaddedBatch) -> torch.Tensor:
        """
        Applies global average pooling to the input tensor x_pb.
        Applies a mask to the input tensor x_pb before averaging.

        x is a Padded batch, where payload is 
        a torch.Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # mask.shape = (batch_size, seq_len)
        mask = x_pb.seq_len_mask
        x = x_pb.payload
        x = x * mask.unsqueeze(-1)
        x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return x


class GlobalAvgPoolAndLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.avg_pool = GlobalAvgPool()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x_pb: PaddedBatch):
        x = self.avg_pool(x_pb)
        return self.linear(x)
    
