from src import utils
import hydra
from omegaconf import DictConfig
import os
# from src.utils import general_helpers
from typing import List

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.logger import Logger

import src.utils.general as utils
from src.utils import hydra_custom_resolvers

log = utils.get_pylogger(__name__)


def run_inference(config: DictConfig):
    # assert config.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    # config.output_dir = general_helpers.get_absolute_path(config.output_dir)
    # log.info(f"Output directory: {config.output_dir}")

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    # print current working directory
    log.info(f"Current working directory: {os.getcwd()}")

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(hydra.utils.get_original_cwd(), ckpt_path)

    log.info(f"Instantiating data module <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, datamodule=datamodule, _recursive_=False)
    
    # Init LIT loggers
    # logger: List[LightningLoggerBase] = []
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
                if lg_conf["_target_"] == 'pytorch_lightning.loggers.wandb.WandbLogger' and config.get("track_gradients",False):
                    logger[-1].watch(model)
    
    callbacks = []

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger, _convert_="partial")

    # Send some parameters from configs to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config= config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    
    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=config.model.checkpoint_path)
    trainer.test(model=model, datamodule=datamodule)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=config.ckpt_path)

    metric_dict = trainer.callback_metrics
    log.info("Metrics dict:")
    log.info(metric_dict)


@hydra.main(config_path="configs", config_name="inference_root", version_base="1.2")
def main(hydra_config: DictConfig):
    import src.utils.general as utils

    utils.extras(hydra_config)
    print(f"Original working directory: {hydra_config.work_dir}")

    # inference on model
    run_inference(hydra_config)


if __name__ == "__main__":
    main()
