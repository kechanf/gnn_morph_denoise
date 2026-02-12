import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 节点 token 级别：预测 one‑hot X 是否等于真实 X
        self.register_buffer("node_correct", torch.zeros(1))
        self.register_buffer("node_total", torch.zeros(1))

        # 边 token 级别：是否预测到“有边”（非 no_edge）
        self.register_buffer("edge_tp", torch.zeros(1))
        self.register_buffer("edge_fp", torch.zeros(1))
        self.register_buffer("edge_fn", torch.zeros(1))
        self.register_buffer("edge_tn", torch.zeros(1))

    @torch.no_grad()
    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        """
        Args:
            masked_pred_X: (bs, n, dx_out)  模型对节点特征的预测（logits / prob）
            masked_pred_E: (bs, n, n, de_out)  模型对边特征的预测
            true_X:        (bs, n, dx_out)  one‑hot 节点特征
            true_E:        (bs, n, n, de_out) one‑hot 边特征，其中 0 通道表示 no_edge
        """
        # ---------- 节点 accuracy ----------
        # 以 true_X 是否全 0 作为“该位置是否为真实节点”的掩码
        valid_nodes = true_X.sum(dim=-1) > 0  # (bs, n)

        if valid_nodes.any():
            true_node_label = true_X.argmax(dim=-1)              # (bs, n)
            pred_node_label = masked_pred_X.argmax(dim=-1)       # (bs, n)
            correct = (pred_node_label == true_node_label) & valid_nodes

            self.node_correct += correct.sum()
            self.node_total += valid_nodes.sum()

        # ---------- 边 accuracy / precision / recall ----------
        # 边的有效位置：两端节点都有效
        valid_nodes = valid_nodes  # (bs, n)
        valid_edges = valid_nodes.unsqueeze(1) & valid_nodes.unsqueeze(2)  # (bs, n, n)

        if valid_edges.any():
            # true: 是否为“有边”（任意非 0 通道为 1）
            true_edge_pos = true_E[..., 1:].sum(dim=-1) > 0  # (bs, n, n)

            # pred: 预测的最大通道是否为非 0（非 no_edge）
            pred_edge_label = masked_pred_E.argmax(dim=-1)   # (bs, n, n)
            pred_edge_pos = pred_edge_label != 0

            # 只统计有效边
            true_edge_pos = true_edge_pos & valid_edges
            pred_edge_pos = pred_edge_pos & valid_edges
            valid = valid_edges

            tp = (pred_edge_pos & true_edge_pos & valid).sum()
            fp = (pred_edge_pos & (~true_edge_pos) & valid).sum()
            fn = ((~pred_edge_pos) & true_edge_pos & valid).sum()
            tn = ((~pred_edge_pos) & (~true_edge_pos) & valid).sum()

            self.edge_tp += tp
            self.edge_fp += fp
            self.edge_fn += fn
            self.edge_tn += tn

    def reset(self):
        self.node_correct.zero_()
        self.node_total.zero_()
        self.edge_tp.zero_()
        self.edge_fp.zero_()
        self.edge_fn.zero_()
        self.edge_tn.zero_()

    def log_epoch_metrics(self):
        eps = torch.finfo(torch.float32).eps

        # 节点预测准确率
        if self.node_total > 0:
            node_acc = (self.node_correct / (self.node_total + eps)).item()
        else:
            node_acc = 0.0

        # 边 token 的二分类指标（有边 vs 无边）
        tp = self.edge_tp
        fp = self.edge_fp
        fn = self.edge_fn
        tn = self.edge_tn
        total = tp + fp + fn + tn

        if total > 0:
            edge_acc = ((tp + tn) / (total + eps)).item()
            edge_prec = (tp / (tp + fp + eps)).item()
            edge_rec = (tp / (tp + fn + eps)).item()
        else:
            edge_acc = edge_prec = edge_rec = 0.0

        node_metrics = {
            "train/node_acc": node_acc,
        }
        edge_metrics = {
            "train/edge_acc": edge_acc,
            "train/edge_precision": edge_prec,
            "train/edge_recall": edge_rec,
        }
        return node_metrics, edge_metrics


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
            """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
            diff = preds - target
            sum_squared_error = torch.sum(diff * diff)
            n_obs = preds.shape[0]
            return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__()
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples