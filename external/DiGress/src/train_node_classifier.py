import os

import hydra
from omegaconf import DictConfig
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from diffusion.extra_features import DummyExtraFeatures
from datasets.morphology_dataset import (
    MorphologyGraphDataModule,
    MorphologyDatasetInfos,
)
from models.node_classifier import NodeClassifier


def load_encoder_from_ckpt(model: NodeClassifier, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # 过滤出 GraphTransformer 的参数（在 DiscreteDenoisingDiffusion 中前缀通常为 'model.'）
    encoder_state = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.encoder.load_state_dict(encoder_state, strict=False)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    独立的节点分类训练脚本：
    - dataset 固定为 morphology（原始抽象形态学图）；
    - general.name 默认改为 node_cls_morphology，便于区分日志/ckpt。
    其它超参（lr、batch_size 等）沿用原有 config。
    """
    cfg.dataset.name = "morphology"
    if cfg.general.name == "debug":
        cfg.general.name = "node_cls_morphology"

    datamodule = MorphologyGraphDataModule(cfg, seed=cfg.train.seed)
    dataset_infos = MorphologyDatasetInfos(datamodule, cfg.dataset)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=DummyExtraFeatures(),
        domain_features=DummyExtraFeatures(),
    )

    model = NodeClassifier(cfg, dataset_infos)

    # 如有需要，可在 cfg.model.encoder_ckpt 中指定 DiGress ckpt 路径
    encoder_ckpt = getattr(cfg.model, "encoder_ckpt", None)
    if encoder_ckpt is not None and os.path.isfile(encoder_ckpt):
        load_encoder_from_ckpt(model, encoder_ckpt)

    callbacks = []
    ckpt_dir = f"checkpoints/{cfg.general.name}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}",
        monitor="val/acc",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,
    )
    last_ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename="last", every_n_epochs=1)
    callbacks.extend([checkpoint_callback, last_ckpt])

    csv_logger = CSVLogger(save_dir=".", name=cfg.general.name, flush_logs_every_n_steps=1)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=[csv_logger],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()


