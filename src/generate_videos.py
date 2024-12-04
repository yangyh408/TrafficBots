import hydra
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from typing import List
from pytorch_lightning import seed_everything, LightningDataModule, LightningModule, Trainer, Callback
from pytorch_lightning.loggers import LightningLoggerBase
import os
from data_modules.data_h5_womd import DataH5womd

def download_checkpoint(loggers, wb_ckpt) -> None:
    if os.environ.get("LOCAL_RANK", 0) == 0:
        artifact = loggers[0].experiment.use_artifact(wb_ckpt, type="model")
        artifact_dir = artifact.download("ckpt")

# 加载配置信息到变量
hydra.initialize(config_path="../configs/")
config = hydra.compose(config_name="run.yaml")

# 设置随机数种子
seed_everything(config.seed, workers=True)

# 组织和管理数据加载
# hydra.utils.instantiate 函数是 Hydra 提供的一个实用函数，用于实例化给定类或模块，并根据配置文件中的参数初始化实例。
datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

callbacks: List[Callback] = []
if "callbacks" in config:
    for _, cb_conf in config.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb_conf))

loggers: List[LightningLoggerBase] = []
if "loggers" in config:
    for _, lg_conf in config.loggers.items():
        loggers.append(hydra.utils.instantiate(lg_conf))

download_checkpoint(loggers, config.resume.checkpoint)
ckpt_path = "ckpt/model.ckpt"
modelClass = hydra.utils.get_class(config.model._target_)
model = modelClass.load_from_checkpoint(
    ckpt_path, wb_artifact=config.resume.checkpoint, **config.resume.model_overrides
)
config.trainer.resume_from_checkpoint = ckpt_path

# model: LightningModule = hydra.utils.instantiate(
#     config.model, data_size=datamodule.tensor_size_train, _recursive_=False
# )

trainer: Trainer = hydra.utils.instantiate(
    config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
)

trainer.validate(model=model, datamodule=datamodule)