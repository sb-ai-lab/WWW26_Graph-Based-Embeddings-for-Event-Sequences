import pytorch_lightning as pl


# This callback is useful when testing `get_models_from_checkpoints.sh` 
# scripts and there are no checkpoints locally.
class SaveCheckpointOnTrainStart(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        checkpoint_path = "checkpoint_at_start.ckpt"
        trainer.save_checkpoint(checkpoint_path)