from typing import Optional

import torch 
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from .avg_pool import GlobalAvgPoolAndLinear


class AvgPoolLinearEncoder(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.avg_pool_linear = GlobalAvgPoolAndLinear(input_size, output_size)

    def forward(self, x_pb: PaddedBatch):
        return self.avg_pool_linear(x_pb)
    
    @property
    def output_size(self) -> int:
        return self.avg_pool_linear.linear.out_features


class AvgPoolLinearSeqEncoder(torch.nn.Module):

    # It's not a child of SeqEncoderContainer because
    # is_reduce_sequence setter and getter don't make sense here
    # and may lead to confusion.

    def __init__(self, trx_encoder: TrxEncoderBase, output_size: Optional[int] = None):
        # super().__init__(
        #     trx_encoder=trx_encoder,
        #     seq_encoder_cls=AvgPoolLinearEncoder,
        #     input_size=in_features,
        #     seq_encoder_params={'output_size': out_features}
        #     is_reduce_sequence=False
        # )
        super().__init__()
        input_size = trx_encoder.output_size
        if output_size is None:
            output_size = input_size
        self.trx_encoder = trx_encoder
        self.seq_encoder = AvgPoolLinearEncoder(trx_encoder.output_size, output_size)

    def forward(self, x):
        x = self.trx_encoder(x)
        return self.seq_encoder(x)
    
    @property
    def category_max_size(self):
        return self.trx_encoder.category_max_size

    @property
    def category_names(self):
        return self.trx_encoder.category_names

    @property
    def embedding_size(self):
        return self.seq_encoder.embedding_size
