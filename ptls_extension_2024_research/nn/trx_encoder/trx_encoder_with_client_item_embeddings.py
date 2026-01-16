import warnings
from typing import Dict, Optional, List, Callable, Tuple

import torch
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.batch_norm import RBatchNorm, RBatchNormWithLens
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase
from ptls.nn.trx_encoder.encoders import BaseEncoder  # for type hinting

from .client_item_encoder import BaseClientItemEncoder



class TrxEncoder_WithCIEmbeddings(TrxEncoderBase):
    """
    A NON-CLIENT-AGNOSTIC Network layer which creates 
    a representation for single transactions.

    Diffrent from ptls.nn.trx_encoder.trx_encoder.TrxEncoder in theese aspects:
    * Input is Tuple[feats_pb, client_ids]
        * feats: PaddedBatch
            contains ptls-format dictionary with feature arrays of shape (B, T)
        * client_ids: torch.Tensor
            has shape (B,) contains a client_id corresponding 
            to the `B` dimension of features in `feats_pb`
    * Has an additional `client_item_embeddings` parameter

    This class was createed to support transaction encoders that are not
    client agnostic.
    Particularly to make a TrxEncoder that uses a client_id and items_ids to
    add embeddings from a GNN.


    Output format is same as in ptls.nn.trx_encoder.trx_encoder.TrxEncoder:
    `PaddedBatch` with transaction embeddings of shape (B, T, H)
    where:
        B - batch size, sequence count in batch
        T - sequence length
        H - hidden size, representation dimension   

    
    
    `ptls.nn.trx_encoder.noisy_embedding.NoisyEmbedding` implementation are used for categorical features.

    Parameters
        embeddings:
            dict with categorical feature names.
            Values must be like this `{'in': dictionary_size, 'out': embedding_size}`
            These features will be encoded with lookup embedding table of shape (dictionary_size, embedding_size)
            Values can be a `torch.nn.Embedding` implementation
        numeric_values:
            dict with numerical feature names.
            Values must be a string with scaler_name.
            Possible values are: 'identity', 'sigmoid', 'log', 'year'.
            These features will be scaled with selected scaler.
            Values can be `ptls.nn.trx_encoder.scalers.BaseScaler` implementatoin

            One field can have many scalers. In this case key become alias and col name should be in scaler.
            Check `TrxEncoderBase.numeric_values` for more details

        custom_embeddings:
            A Dict where a key is a feature name and a value is an embedder. 
            An embedder is an object of a class that must 
            inherit from ptls.nn.trx_encoder.encoders.BaseEncoder 
            and define output_size property.
        
        col_item_ids:
            Column name in input padded batch that contains item ids.
            This column will be used to get item embeddings from `client_item_embeddings`
        client_item_embeddings:
            List of `BaseClientItemEncoder` objects.

        embeddings_noise (float):
            Noise level for embedding. `0` meens without noise
        emb_dropout (float):
            Probability of an element of embedding to be zeroed
        spatial_dropout (bool):
            Whether to dropout full dimension of embedding in the whole sequence

        use_batch_norm:
            True - All numerical values will be normalized after scaling
            False - No normalizing for numerical values
        use_batch_norm_with_lens:
            True - Respect seq_lens during batch_norm. Padding zeroes will be ignored
            False - Batch norm ower all time axis. Padding zeroes will included.

        orthogonal_init:
            if True then `torch.nn.init.orthogonal` applied
        linear_projection_size:
            Linear layer at the end will be added for non-zero value

        out_of_index:
            How to process a categorical indexes which are greater than dictionary size.
            'clip' - values will be collapsed to maximum index. This works well for frequency encoded categories.
                We join infrequent categories to one.
            'assert' - raise an error of invalid index appear.

        norm_embeddings: keep default value for this parameter
        clip_replace_value: Not useed. keep default value for this parameter
        positions: Not used. Keep default value for this parameter
    """

    def __init__(self,
                 embeddings: Optional[Dict[str, Dict[str, int]]] = None,
                 numeric_values: Optional[Dict[str, str]] = None,
                 custom_embeddings: Optional[Dict[str, BaseEncoder]] = None,
                 col_item_ids: str = None,
                 client_item_embeddings: Optional[List[BaseClientItemEncoder]] = None,
                 embeddings_noise: float = 0,
                 norm_embeddings=None,
                 use_batch_norm=True,
                 use_batch_norm_with_lens=False,
                 clip_replace_value=None,
                 positions=None,
                 emb_dropout=0,
                 spatial_dropout=False,
                 orthogonal_init=False,
                 linear_projection_size=0,
                 out_of_index: str = 'clip',
                 ) -> None:
        if client_item_embeddings is not None:
            assert col_item_ids is not None, 'col_item_ids must be provided if client_item_embeddings is not None'
        
        self.col_item_ids = col_item_ids
              
        if clip_replace_value is not None:
            warnings.warn('`clip_replace_value` attribute is deprecated. Always "clip to max" used. '
                          'Use `out_of_index="assert"` to avoid categorical values clip', DeprecationWarning)

        if positions is not None:
            warnings.warn('`positions` is deprecated. positions is not used', UserWarning)

        if embeddings is None:
            embeddings = {}
        if custom_embeddings is None:
            custom_embeddings = {}


        noisy_embeddings = {}
        for emb_name, emb_props in embeddings.items():
            if emb_props.get('disabled', False):
                continue
            if emb_props['in'] == 0 or emb_props['out'] == 0:
                continue
            noisy_embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=1 if norm_embeddings else None,
                noise_scale=embeddings_noise,
                dropout=emb_dropout,
                spatial_dropout=spatial_dropout,
            )

        super().__init__(
            embeddings=noisy_embeddings,
            numeric_values=numeric_values,
            custom_embeddings=custom_embeddings,
            out_of_index=out_of_index,
        )


        if client_item_embeddings is None:
            client_item_embeddings = []
        self.client_item_embeddings = torch.nn.ModuleList(client_item_embeddings)

        custom_embedding_size = self.custom_embedding_size
        if use_batch_norm and custom_embedding_size > 0:
            # :TODO: Should we use Batch norm with not-numerical custom embeddings?
            if use_batch_norm_with_lens:
                self.custom_embedding_batch_norm = RBatchNormWithLens(custom_embedding_size)
            else:
                self.custom_embedding_batch_norm = RBatchNorm(custom_embedding_size)
        else:
            self.custom_embedding_batch_norm = None

        if linear_projection_size > 0:
            self.linear_projection_head = torch.nn.Linear(super().output_size, linear_projection_size)
        else:
            self.linear_projection_head = None

        if orthogonal_init:
            for n, p in self.named_parameters():
                if n.startswith('embeddings.') and n.endswith('.weight'):
                    torch.nn.init.orthogonal_(p.data[1:])
                if n == 'linear_projection_head.weight':
                    torch.nn.init.orthogonal_(p.data)

    def _get_custom_embeddings_tensor(self, x: PaddedBatch) -> Optional[torch.Tensor]:
        """
        Get custom embeddings tensor that is a concatenation of all custom embeddings.
        Optionally apply batch norm to the custom embeddings tensor.
        """
        if not self.custom_embeddings:
            return None
        
        processed_custom_embeddings = []
        for field_name in self.custom_embeddings.keys():
            # processed_custom_embeddings[i].shape = (B, T, H_embeder_i)
            processed_custom_embeddings.append(self.get_custom_embeddings(x, field_name))

        custom_embeddings_tensor = torch.cat(processed_custom_embeddings, dim=2)

        if self.custom_embedding_batch_norm is not None:
            custom_embeddings_tensor_pb = PaddedBatch(custom_embeddings_tensor, x.seq_lens)
            custom_embeddings_tensor_pb = self.custom_embedding_batch_norm(custom_embeddings_tensor_pb)
            custom_embeddings_tensor = custom_embeddings_tensor_pb.payload

        # custom_embeddings_tensor.shape = (B, T, H_embeder_0 + H_embeder_1 + ... H_embeder_n)
        return custom_embeddings_tensor
    

    def _get_client_item_embeddings_lst(self, 
                                        item_ids: torch.Tensor,
                                        client_ids: torch.Tensor) -> List[torch.Tensor]:
        """
        Get client item embeddings tensor that is a concatenation of all client item embeddings.
        """
        processed_client_item_embeddings_lst = []
        for embedder in self.client_item_embeddings:
            # processed_client_item_embeddings[i].shape = (B, T, H_ci_embeder_i)
            processed_client_item_embeddings_lst.append(embedder(client_ids, item_ids))
        return processed_client_item_embeddings_lst
            

    def forward(self, feats_and_client_ids: Tuple[PaddedBatch, torch.Tensor]):
        feats_pb, client_ids = feats_and_client_ids

        if self.client_item_embeddings:
            assert self.col_item_ids in feats_pb.payload, f'Item ids column `{self.col_item_ids}` not found in feats_pb'


        processed_embeddings = []

        for field_name in self.embeddings.keys():
            # processed_embeddings[i].shape = (B, T, H_embed_layer)
            processed_embeddings.append(self.get_category_embeddings(feats_pb, field_name))
        
        custom_embeddings_tensor = self._get_custom_embeddings_tensor(feats_pb)
        if custom_embeddings_tensor is not None:
            # custom_embeddings_tensor.shape = (B, T, H_embeder_0 + H_embeder_1 + ... H_embeder_n)
            processed_embeddings.append(custom_embeddings_tensor)

        if self.client_item_embeddings:
            item_ids = feats_pb.payload[self.col_item_ids]
            processed_client_item_embeddings_lst = self._get_client_item_embeddings_lst(item_ids, client_ids)
            processed_embeddings.extend(processed_client_item_embeddings_lst)
        
        # for emb in processed_embeddings:
        #     print(emb.shape)
        
        
        out = torch.cat(processed_embeddings, dim=2)

        if self.linear_projection_head is not None:
            out = self.linear_projection_head(out)
        
        return PaddedBatch(out, feats_pb.seq_lens)
    

    @property
    def client_item_embedding_size(self):
        return sum(e.output_size for e in self.client_item_embeddings)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.linear_projection_head is not None:
            return self.linear_projection_head.out_features
        return self.embedding_size + self.custom_embedding_size + self.client_item_embedding_size
