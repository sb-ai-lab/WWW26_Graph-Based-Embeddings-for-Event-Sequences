import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .pl_train_module_utils import get_git_commit_hash 

logger = logging.getLogger(__name__)


@hydra.main(version_base='1.2', config_path=None, config_name=None)
def main(conf: DictConfig):
    # print("config:\n")
    # print(OmegaConf.to_yaml(conf))
    # print("config_end\n\n\n")

    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)

    model = hydra.utils.instantiate(conf.pl_module)

    if 'model_weights_only_ckpt' in conf:
        model_weights_only_ckpt = torch.load(conf['model_weights_only_ckpt'])
        model.load_state_dict(model_weights_only_ckpt['state_dict'])

    dm = hydra.utils.instantiate(conf.data_module)

    _trainer_params = conf.trainer
    _trainer_params_additional = {}
    _use_best_epoch = _trainer_params.get('use_best_epoch', False)

    if 'callbacks' in _trainer_params:
        logger.warning(f'Overwrite `trainer.callbacks`, was `{_trainer_params.get("enable_checkpointing", _trainer_params.get("checkpoint_callback", None))}`')
    _trainer_params_callbacks = []

    if _use_best_epoch:
        checkpoint_callback = ModelCheckpoint(monitor=model.metric_name, mode='max')
        logger.info(f'Create ModelCheckpoint callback with monitor="{model.metric_name}"')
        _trainer_params_callbacks.append(checkpoint_callback)

    if _trainer_params.get('checkpoints_every_n_val_epochs', False):
        print(f"{dir(_trainer_params) = }")
        every_n_val_epochs = _trainer_params.checkpoints_every_n_val_epochs
        dirpath = _trainer_params.checkpoint_dirpath
        filename = _trainer_params.checkpoint_filename

        del _trainer_params.checkpoint_dirpath
        del _trainer_params.checkpoint_filename

        checkpoint_callback = ModelCheckpoint(dirpath=dirpath, filename=filename, every_n_epochs=every_n_val_epochs, save_top_k=-1)
        logger.info(f'Create ModelCheckpoint callback every_n_epochs ="{every_n_val_epochs}"')
        _trainer_params_callbacks.append(checkpoint_callback)

        if 'checkpoint_callback' in _trainer_params:
            del _trainer_params.checkpoint_callback
        if 'enable_checkpointing' in _trainer_params:
            del _trainer_params.enable_checkpointing
        del _trainer_params.checkpoints_every_n_val_epochs

    if 'logger_name' in conf:
        _trainer_params_additional['logger'] = TensorBoardLogger(
            save_dir='lightning_logs',
            name=conf.get('logger_name'),
        )
    if not isinstance(_trainer_params.get('strategy', ''), str): # if strategy not exist or str do nothing, 
        _trainer_params_additional['strategy'] = hydra.utils.instantiate(_trainer_params.strategy)
        del _trainer_params.strategy

    if 'additional_callbacks' in _trainer_params:
        for additional_callback_params in _trainer_params['additional_callbacks']:
            _trainer_params_callbacks.append(
                hydra.utils.instantiate(additional_callback_params))
        del _trainer_params.additional_callbacks

    lr_monitor = LearningRateMonitor(logging_interval='step')
    _trainer_params_callbacks.append(lr_monitor)

    if len(_trainer_params_callbacks) > 0:
        _trainer_params_additional['callbacks'] = _trainer_params_callbacks

    resume_checkpoint_path = None 
    if 'resume_checkpoint_path' in _trainer_params:
        resume_checkpoint_path = _trainer_params['resume_checkpoint_path']
        del _trainer_params.resume_checkpoint_path

    print(f"{_trainer_params = }")
    print(f"{_trainer_params_additional = }")

    trainer = pl.Trainer(**_trainer_params, **_trainer_params_additional)
    trainer.fit(model, dm, ckpt_path=resume_checkpoint_path)

    if 'model_path' in conf:
        if _use_best_epoch:
            # from shutil import copyfile
            # copyfile(checkpoint_callback.best_model_path, conf.model_path)
            model.load_from_checkpoint(checkpoint_callback.best_model_path)
            torch.save(model.seq_encoder, conf.model_path)
            logging.info(f'Best model stored in "{checkpoint_callback.best_model_path}" '
                         f'and copied to "{conf.model_path}"')
        else:
            torch.save(model.seq_encoder, conf.model_path)
            logger.info(f'Model weights saved to "{conf.model_path}"')
        
        if 'additional_artifacts_to_save' in conf:
            additional_artifacts = {}
            if 'git_commit_hash' in conf['additional_artifacts_to_save']:
                additional_artifacts['git_commit_hash'] = get_git_commit_hash()
            if 'full_pl_module' in conf['additional_artifacts_to_save']:
                additional_artifacts['full_pl_module'] = model
        
            torch.save(additional_artifacts, conf.model_path + '_additional_artifacts.pt')


if __name__ == '__main__':
    main()
