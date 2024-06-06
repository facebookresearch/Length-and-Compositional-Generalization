from typing import List, Optional

import hydra
import os
from omegaconf import DictConfig
import omegaconf
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
# from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.logger import Logger
import json
import torch

import src.utils.general as utils

log = utils.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """Contains the training pipeline.
    Instantiates all PyTorch Lightning objects from configs.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score useful for hyperparameter optimization.
    """

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    use_ddp = config.trainer.get('strategy', False)
    print("use_ddp",use_ddp)

    torch.set_float32_matmul_precision('medium') # 'high'

    # print current working directory
    log.info(f"Current working directory: {os.getcwd()}")

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(hydra.utils.get_original_cwd(), ckpt_path)

    # Initialize the LIT data module
    log.info(f"Instantiating data module <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, use_ddp=use_ddp, _recursive_=False)

    # Initialize the LIT model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, datamodule=datamodule, dataset_parameters=datamodule.dataset_parameters, _recursive_=False)

    # Initialize LIT callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

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

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from configs to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Print the path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best checkpoint at: {trainer.checkpoint_callback.best_model_path}")
        # log.info(f"Best checkpoint Directory: {os.path.dirname(trainer.checkpoint_callback.best_model_path)}")
        # log.info(f"Best checkpoint filename: {os.path.basename(trainer.checkpoint_callback.best_model_path)}")
        with open("best_ckpt_path.txt", "w") as f:
            f.write(os.path.basename(trainer.checkpoint_callback.best_model_path))

    # Test the model
    if config.get("test"):
        ckpt_path = "best"  # Use the best checkpoint from the previous trainer.fit() call
        _model = None

        if config.get("ckpt_path"):
            ckpt_path = config.get("ckpt_path")  # Use the checkpoint passed in the config
        # TODO: check if this is necessary, it doesn't seem to be
        elif not config.get("train") or config.trainer.get("fast_dev_run"):
            _model = model
            ckpt_path = None  # Use the passed model as it is

        log.info("Starting testing!")
        trainer.test(model=_model, datamodule=datamodule, ckpt_path=ckpt_path)
        metric_dict = trainer.callback_metrics
        log.info("Metrics dict:")
        log.info(metric_dict)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )

    # # write to json, an empty file for now.
    # # TODO: write metrics in test step of all models that don't return any
    # with open("metrics.json", "w") as f:
    #     # converting tensor typed objects to numpy arrays and then to lists so they
    #     # can be serialized
    #     for key, value in trainer.callback_metrics.items():
    #         trainer.callback_metrics[key] = value.cpu().numpy().tolist()

    #     f.write(json.dumps(trainer.callback_metrics))
    # score = trainer.callback_metrics.get(optimized_metric)
    # return score
    return
