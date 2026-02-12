import glob
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import subgraph

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


def _build_target_subgraph(orig: Data) -> Tuple[Data, torch.Tensor]:
    """
    根据 orig.y==1 的节点子集，构造“目标树”子图。
    不修改 orig，只返回一个新的 Data。
    """
    if not hasattr(orig, "y") or orig.y is None:
        raise ValueError("Expected orig.y to contain node labels (0/1) for denoising task.")

    node_labels = orig.y.view(-1)
    if node_labels.dtype != torch.long:
        node_labels = node_labels.long()

    mask = node_labels == 1  # 1 表示目标树节点
    if mask.sum() == 0:
        # 极端情况：没有目标节点，退化为整图作为 target
        mask = torch.ones_like(mask, dtype=torch.bool)

    # 使用 PyG 的 subgraph 工具提取子图并重排索引
    edge_index, edge_attr = getattr(orig, "edge_index"), getattr(orig, "edge_attr", None)
    sub_edge_index, sub_edge_attr = subgraph(mask, edge_index, edge_attr=edge_attr, relabel_nodes=True)

    # 节点特征：这里保持与抽象图一致，使用全 1
    num_target_nodes = mask.sum().item()
    x_target = torch.ones(num_target_nodes, 1, dtype=torch.float)

    # edge_attr：如果原来没有 edge_attr，就创建 [no_edge, edge] one-hot；有的话只保留“有边”这一类
    if sub_edge_attr is None:
        num_edges = sub_edge_index.size(1)
        sub_edge_attr = torch.zeros(num_edges, 2, dtype=torch.float)
        sub_edge_attr[:, 1] = 1.0

    y_global = torch.zeros(1, 0).float()
    n_nodes = torch.tensor([num_target_nodes], dtype=torch.long)

    target = Data(
        x=x_target,
        edge_index=sub_edge_index,
        edge_attr=sub_edge_attr,
        y=y_global,
        n_nodes=n_nodes,
    )
    return target, mask


class MorphologyDenoiseDataset(Dataset):
    """
    成对数据集：
    - 条件图 G_noise：完整的“噪声森林”；
    - 目标图 G_target：由 orig.y==1 的节点诱导的“干净目标树”子图。

    当前 DiGress 仍然只在 G_target 上训练（无条件抽象生成），
    但我们会在 Data 中保留 G_noise 以便后续实现条件扩散。
    """

    def __init__(self, root_dir: str):
        super().__init__(root_dir, transform=None, pre_transform=None)
        self.root_dir = root_dir
        self.pt_files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        if len(self.pt_files) == 0:
            raise FileNotFoundError(f"No .pt files found under {root_dir}")

    def len(self) -> int:
        return len(self.pt_files)

    def get(self, idx: int) -> Data:
        path = self.pt_files[idx]
        try:
            orig: Data = torch.load(path, weights_only=False)
        except TypeError:
            orig: Data = torch.load(path)

        # 条件图：噪声森林（抽象为单一节点类型 + 单一边类型）
        if hasattr(orig, "num_nodes") and orig.num_nodes is not None:
            num_nodes = int(orig.num_nodes)
        else:
            num_nodes = orig.x.size(0)

        x_noise = torch.ones(num_nodes, 1, dtype=torch.float)
        edge_index_noise = orig.edge_index
        num_edges_noise = edge_index_noise.size(1)
        edge_attr_noise = torch.zeros(num_edges_noise, 2, dtype=torch.float)
        edge_attr_noise[:, 1] = 1.0

        # 目标图：由 orig.y==1 的节点诱导
        target, mask = _build_target_subgraph(orig)

        # 在返回的 Data 中既包含 target（作为当前训练目标），也缓存 noise 供后续条件模型使用
        data = target
        data.x_noise = x_noise
        data.edge_index_noise = edge_index_noise
        data.edge_attr_noise = edge_attr_noise
        data.noise_n_nodes = torch.tensor([num_nodes], dtype=torch.long)
        data.orig_x = getattr(orig, "x", None)
        data.orig_y = getattr(orig, "y", None)
        data.noise_mask = ~mask  # 噪声节点掩码（在原图索引空间）

        return data


class MorphologyDenoiseDataModule(AbstractDataModule):
    """
    使用 MorphologyDenoiseDataset 的 DataModule，按照 0.8 / 0.1 / 0.1 划分 train/val/test。
    """

    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        datadir = cfg.dataset.datadir

        if not os.path.isabs(datadir):
            base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            datadir = os.path.join(base_path, datadir)

        if not os.path.isdir(datadir):
            raise FileNotFoundError(f"Morphology datadir not found: {datadir}")

        full_dataset = MorphologyDenoiseDataset(datadir)

        n_total = len(full_dataset)
        n_train = int(round(n_total * 0.8))
        n_val = int(round((n_total - n_train) * 0.5))
        n_test = n_total - n_train - n_val

        if n_train == 0 or n_val == 0 or n_test == 0:
            raise ValueError(
                f"Not enough graphs ({n_total}) to split into train/val/test. "
                f"Consider generating more samples."
            )

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

        datasets: Dict[str, Dataset] = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class MorphologyDenoiseDatasetInfos(AbstractDatasetInfos):
    """
    针对 denoise 任务的 DatasetInfos：
    - 统计目标树图的节点数 / 边类型分布等；
    - 条件图的统计留待后续条件扩散实现时再用。
    """

    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = "morphology_denoise"

        self.n_nodes = self.datamodule.node_counts(max_nodes_possible=5000)
        self.node_types = torch.tensor([1.0])
        self.edge_types = self.datamodule.edge_counts()

        super().complete_infos(self.n_nodes, self.node_types)

