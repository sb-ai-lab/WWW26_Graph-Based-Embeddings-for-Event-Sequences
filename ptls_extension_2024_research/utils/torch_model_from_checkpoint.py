import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path=None, config_name=None)
def main(conf: DictConfig):
    ckpt = torch.load(conf.ckpt_path, map_location=conf.device)
    model: pl.LightningModule = hydra.utils.instantiate(conf.pl_module)
    model.load_state_dict(ckpt['state_dict'])
    torch.save(model.seq_encoder, conf.model_path)

if __name__ == '__main__':
    main()
