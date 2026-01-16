from typing import List

import pytorch_lightning as pl

from .model_modifiers import ModelModifier


class ModifyModelAfterNEpochesCallback(pl.Callback):
    def __init__(self, 
                 model_modifier: ModelModifier,
                 n_epoches_before_change: int) -> None:
        super().__init__()
        self._model_modifier = model_modifier
        self._n_epoches_before_change = n_epoches_before_change

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch == self._n_epoches_before_change:
           self._model_modifier(pl_module)


class ModifyModelAfterNBatchesCallback(pl.Callback):
    def __init__(self, 
                 model_modifier: ModelModifier,
                 n_batches_before_change: int) -> None:
        super().__init__()
        self._model_modifier = model_modifier
        self._n_batches_before_change = n_batches_before_change

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if trainer.global_step == self._n_batches_before_change:
           self._model_modifier(pl_module)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        checkpoint['pl_module'] = pl_module
        return checkpoint
