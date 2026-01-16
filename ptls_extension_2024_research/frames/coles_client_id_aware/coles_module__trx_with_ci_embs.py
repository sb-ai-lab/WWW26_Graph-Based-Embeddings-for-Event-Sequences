from typing import Tuple

import torch  # for typing
from ptls.data_load.padded_batch import PaddedBatch  # for typing
from ptls.frames.coles import CoLESModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from ptls_extension_2024_research.lightning_utlis import LogLstEl


class CoLESModule_CITrx(CoLESModule):
    """
    Same as ptls.frames.coles.CoLESModule, except 
    TrxEncoder_WithClientItemEmbeddings is used as trx_encoder and thus 
    it takes a tuple:
    (`padded_batch_of_dict_with_seq_feats`, `client_ids`)
    instead of just `padded_batch_of_dict_with_seq_feats`
    """
    # def __init__(self,
    #              seq_encoder: SeqEncoderContainer = None,
    #              head=None,
    #              loss=None,
    #              validation_metric=None,
    #              optimizer_partial=None,
    #              lr_scheduler_partial=None,
    #              is_return_instead_log = False):
    #     super().__init__(seq_encoder = seq_encoder,
    #                     head=head,
    #                     loss=loss,
    #                     validation_metric=validation_metric,
    #                     optimizer_partial=optimizer_partial,
    #                     lr_scheduler_partial=lr_scheduler_partial)
    #     self.is_return_instead_log = is_return_instead_log

    def shared_step(self, x: PaddedBatch, client_ids: torch.Tensor):
        y_h = self((x, client_ids))
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, client_ids
    
    def get_seq_len_log_lst_el(self, batch: Tuple[PaddedBatch, torch.Tensor]) -> LogLstEl:
        if type(batch) is tuple:
            x, y = batch
            if isinstance(x, PaddedBatch):
                return LogLstEl(
                    'seq_len', x.seq_lens.float().mean(), [], {'prog_bar':True})
        else:
            # log_lst_el = LogLstEl('seq_len', -1, [], {'prog_bar':True})
            # this code should not be reached
            raise AssertionError('batch is not a tuple')


    def _training_step(self, batch, batch_idxs):
        log_lst = []

        y_h, y = self.shared_step(*batch)
        loss = self._loss(y_h, y)
        log_lst.append(LogLstEl('loss', loss, [], {}))
        log_lst.append(self.get_seq_len_log_lst_el(batch))
        
        
        return (loss, log_lst)


    # temp solution to avoid logging outside callbacks in coles_gnn_modules
    def training_step(self, batch, batch_idxs):
        loss, log_lst = self._training_step(batch, batch_idxs)
        for el in log_lst:
            self.log(el.name, el.value, *el.args, **el.kwargs)
        return loss
