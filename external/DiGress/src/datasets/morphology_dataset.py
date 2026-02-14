import glob
import os
from typing import Dict

import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class MorphologyGraphDataset(Dataset):
    """
    从目录中读取形态学图（本项目合成的 .pt），并转换为 DiGress 期望的抽象图格式：

    - 只建模拓扑结构，不使用原始几何 / 类型特征与节点标签；
    - 节点特征 X: 全 1（[num_nodes, 1]），表示单一节点类型；
    - 边特征 edge_attr: one-hot(2) = [no_edge, edge]，实际只用到“edge”这一类；
    - 全局 y: 空张量 shape = [1, 0]，表示无图级属性；
    - 额外字段 n_nodes: 图中节点数量，用于节点数分布估计。

    注意：这里不会改变原始 .pt 文件，只是在加载时按 DiGress 的抽象图接口重新封装。
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
        # 适配 PyTorch>=2.6 默认 weights_only=True 导致自定义类无法反序列化的问题。
        # 我们只在本地受信任数据上使用，因此显式关闭 weights_only。
        try:
            orig: Data = torch.load(path, weights_only=False)
        except TypeError:
            # 兼容较旧版本 PyTorch（没有 weights_only 参数）
            orig: Data = torch.load(path)

        # 推断节点数
        if hasattr(orig, "num_nodes") and orig.num_nodes is not None:
            num_nodes = int(orig.num_nodes)
        else:
            num_nodes = orig.x.size(0)

        # 节点特征：单一类型 → 全 1
        x = torch.ones(num_nodes, 1, dtype=torch.float)

        # 边索引：复用原始 edge_index（假定是无向图或已经按需求处理好）
        edge_index = orig.edge_index

        # 边特征：2 维 one-hot，[no_edge, edge]，这里只用 “edge” 这一类
        num_edges = edge_index.size(1)
        edge_attr = torch.zeros(num_edges, 2, dtype=torch.float)
        edge_attr[:, 1] = 1.0

        # 全局属性 y：为空（DiGress 用作条件时才需要）
        y = torch.zeros(1, 0).float()

        # n_nodes：图中节点数（用于节点数分布建模）
        n_nodes = torch.tensor([num_nodes], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            n_nodes=n_nodes,
        )

        # 可选：保留原始特征与标签，便于后续分析 / 可视化
        data.orig_x = getattr(orig, "x", None)
        data.orig_y = getattr(orig, "y", None)

        return data


class MorphologyGraphDataModule(AbstractDataModule):
    """
    使用 MorphologyGraphDataset 的 DataModule，按照 0.8 / 0.1 / 0.1 划分 train/val/test。
    """

    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        datadir = cfg.dataset.datadir

        if not os.path.isabs(datadir):
            # 若给的是相对路径，则相对 DiGress 仓库根目录解析
            base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            datadir = os.path.join(base_path, datadir)

        if not os.path.isdir(datadir):
            raise FileNotFoundError(f"Morphology datadir not found: {datadir}")

        full_dataset = MorphologyGraphDataset(datadir)

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


class MorphologyDatasetInfos(AbstractDatasetInfos):
    """
    为形态学抽象图构建节点数分布 / 节点类型分布 / 边类型分布等统计信息。
    """

    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = "morphology"

        # 节点数分布：本任务的图节点数可能较大，显式放宽 max_nodes_possible。
        # 为避免越界，这里设置为一个比较宽松的上界。
        self.n_nodes = self.datamodule.node_counts(max_nodes_possible=5000)
        # 只有 1 种节点类型（占位），edge_types 由 edge_attr one-hot 决定
        self.node_types = torch.tensor([1.0])
        self.edge_types = self.datamodule.edge_counts()

        super().complete_infos(self.n_nodes, self.node_types)

