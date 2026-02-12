import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.transformer_model import GraphTransformer
from src import utils


class NodeClassifier(pl.LightningModule):
    """
    利用 DiGress 的 GraphTransformer 作为编码器，在其节点 embedding 上
    加一个轻量的监督分类头，对 orig_y 做节点二分类。

    - encoder: 复用 GraphTransformer 结构及其预训练权重；
    - head: MLP(dx -> dx -> 1)，输出每个节点的 logit；
    - loss/metrics: 标准二分类 BCE + 节点级 acc / precision / recall。
    """

    def __init__(self, cfg, dataset_infos):
        super().__init__()
        self.cfg = cfg

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        self.encoder = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        dx = cfg.model.hidden_dims["dx"]
        self.head = nn.Sequential(
            nn.Linear(dx, dx),
            nn.ReLU(),
            nn.Linear(dx, 1),
        )

        fixed_pos_weight = float(getattr(cfg.train, "pos_weight", 1.0))
        self.register_buffer("pos_weight", torch.tensor(max(fixed_pos_weight, 1.0), dtype=torch.float32))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.cls_threshold = float(getattr(cfg.train, "cls_threshold", 0.5))
        self.freeze_encoder_epochs = int(getattr(cfg.train, "freeze_encoder_epochs", 0))

        # 统计指标
        self.register_buffer("cls_correct", torch.zeros(1))
        self.register_buffer("cls_total", torch.zeros(1))
        self.register_buffer("cls_tp", torch.zeros(1))
        self.register_buffer("cls_fp", torch.zeros(1))
        self.register_buffer("cls_fn", torch.zeros(1))

    def reset_metrics(self) -> None:
        self.cls_correct.zero_()
        self.cls_total.zero_()
        self.cls_tp.zero_()
        self.cls_fp.zero_()
        self.cls_fn.zero_()

    def forward(self, data):
        """
        返回 batch 中所有有效节点的 embedding。
        """
        dense, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense = dense.mask(node_mask)
        X, E, y_global = dense.X, dense.E, data.y
        # 形态学图 data.y 可能为 (batch, 0)，encoder 期望至少 1 维（如 time conditioning），补齐
        if y_global.dim() >= 2 and y_global.size(-1) == 0:
            y_global = torch.zeros(y_global.size(0), 1, device=y_global.device, dtype=y_global.dtype)

        node_emb = self.encoder.forward_node_embedding(X, E, y_global, node_mask)  # (bs, n, dx)

        bs, n, dx = node_emb.shape
        node_emb_flat = node_emb.view(bs * n, dx)
        node_mask_flat = node_mask.view(bs * n)
        node_emb_flat = node_emb_flat[node_mask_flat]

        return node_emb_flat

    def _step(self, data, stage: str):
        if not hasattr(data, "orig_y") or data.orig_y is None:
            raise ValueError("NodeClassifier expects data.orig_y to be present.")

        node_emb = self(data)
        labels = data.orig_y.view(-1).float().to(node_emb.device)

        # 对齐长度（理论上应一致，保险起见截断）
        if labels.shape[0] != node_emb.shape[0]:
            min_len = min(labels.shape[0], node_emb.shape[0])
            labels = labels[:min_len]
            node_emb = node_emb[:min_len]

        logits = self.head(node_emb).squeeze(-1)
        loss = self.loss_fn(logits, labels)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.cls_threshold).long()
            true = labels.long()

            correct = (preds == true).sum().to(self.cls_correct.device)
            self.cls_correct += correct
            self.cls_total += torch.tensor(preds.numel(), device=self.cls_total.device, dtype=self.cls_total.dtype)

            tp = ((preds == 1) & (true == 1)).sum().to(self.cls_tp.device)
            fp = ((preds == 1) & (true == 0)).sum().to(self.cls_fp.device)
            fn = ((preds == 0) & (true == 1)).sum().to(self.cls_fn.device)

            self.cls_tp += tp
            self.cls_fp += fp
            self.cls_fn += fn

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, data, batch_idx):
        return self._step(data, stage="train")

    def validation_step(self, data, batch_idx):
        return self._step(data, stage="val")

    def on_train_epoch_start(self) -> None:
        if self.current_epoch < self.freeze_encoder_epochs:
            self.encoder.requires_grad_(False)
        else:
            self.encoder.requires_grad_(True)
        self.reset_metrics()

    def on_validation_epoch_start(self) -> None:
        self.reset_metrics()

    def _log_epoch_metrics(self, stage: str):
        if self.cls_total.item() == 0:
            return
        acc = (self.cls_correct / self.cls_total).item()
        tp = self.cls_tp.item()
        fp = self.cls_fp.item()
        fn = self.cls_fn.item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        self.log(f"{stage}/acc", acc, prog_bar=True)
        self.log(f"{stage}/precision", prec)
        self.log(f"{stage}/recall", rec)

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )

