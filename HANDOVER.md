## 当前项目交接说明（gnn_project）

### 1. 总体目标与当前进度

- **项目目标**: 基于合成树形数据集（morphology）执行 DiGress 两阶段训练 + Graph-Mamba 节点分类（含 Mamba-SSM 完整版本）。
- **当前进度**:
  - 本机 `graph-mamba` 环境中已成功安装并验证：
    - PyTorch **2.8.0+cu128**
    - `torch_geometric 2.0.4` 及其扩展 `torch_scatter/torch_sparse`（已重新编译，GLIBC 问题已解决）
    - `mamba-ssm 2.3.0`（可正常 `import mamba_ssm`）
  - Graph-Mamba **在 CPU 上**可完整跑通 tiny50 的 EX 配置（含 Mamba），但由于 **GPU 驱动仅支持 CUDA 11.4**，无法在本机使用 GPU 跑 Mamba CUDA kernel。
  - tiny50 的 GatedGCN-only 流程（`--no-mamba`）在本机可以正常训练。

### 2. 数据与路径约定

- **统一数据根目录环境变量**: 通过 `GNN_DATA_ROOT` 控制所有数据路径。
  - 推荐在 `~/.bashrc` 中设置：
    ```bash
    export GNN_DATA_ROOT=/PBshare/SEU-ALLEN/Users/KaifengChen/gnn_data/morphology_seg
    ```
  - 若未设置，`config.py` 默认使用：
    ```text
    DATA_ROOT = "/data2/kfchen/tracing_ws/morphology_seg"
    ```
- **关键路径（由 `config.py` 推导）**:
  - **训练数据（full）**: `TRAIN_DATA_DIR = os.path.join(DATA_ROOT, "synthesis_data")`
  - **tiny50 数据**: `TRAIN_DATA_DIR_TINY_50 = os.path.join(DATA_ROOT, "synthesis_data_tiny_50")`
  - **Graph-Mamba 输出**: `GRAPH_MAMBA_OUT_DIR = os.path.join(DATA_ROOT, "graph_mamba_results")`
- **当前实际数据位置**:
  - 已在网络盘上准备好统一存放目录：
    ```text
    /PBshare/SEU-ALLEN/Users/KaifengChen/gnn_data/morphology_seg
    ├── synthesis_data
    └── synthesis_data_tiny_50   # tiny50 实验主要使用
    ```
  - 两台机器都可以访问该网络盘，只要设置好 `GNN_DATA_ROOT`，代码层面无需改路径。

### 3. 环境与脚本说明（Graph-Mamba 相关）

- **Conda 环境**: `graph-mamba`
  - 当前本机状态：
    - `torch 2.8.0+cu128`（CUDA 12.8，GPU 驱动过旧，实际运行在 CPU）
    - `torch_geometric 2.0.4`
    - `torch_scatter / torch_sparse / pyg_lib` 已对齐到 `torch 2.8.0+cu128`
    - `mamba-ssm 2.3.0` 已成功从源码构建
  - 若需从头重建，推荐使用脚本：
    ```bash
    cd /home/kfchen/gnn_project
    bash scripts/setup_graph_mamba_from_requirements.sh graph-mamba
    ```
    该脚本基于 `external/Graph-Mamba/requirements_conda.txt`，默认安装 PyTorch 2.0.0 + CUDA 11.7、PyG 2.0.4、mamba-ssm 等（如需兼容新驱动，可在新机器上优先使用这一方案）。

- **Graph-Mamba 运行脚本**: `scripts/run_graph_mamba.py`
  - 默认配置：
    - **完整 EX（含 Mamba）**: `configs/Mamba/morphology-node-EX.yaml`
    - **GatedGCN-only（无需 Mamba）**: `configs/Mamba/morphology-node-GatedGCN-only.yaml`
  - 关键参数：
    - `--data_dir`: 指定 `.pt` 图数据目录，默认使用 `config.TRAIN_DATA_DIR`
    - `--no-mamba`: 使用 GatedGCN-only 配置，完全不依赖 mamba-ssm
    - `--name_tag`: 输出目录名后缀，方便区分不同实验

- **tiny50 上的完整 Graph-Mamba（含 Mamba）脚本**:  
  `scripts/run_graph_mamba_tiny50_full.sh`
  - 当前内容（要点）：
    - 数据目录：优先使用环境变量 `DATADIR_TINY50`，否则从 `config.TRAIN_DATA_DIR_TINY_50` 读取
    - 环境：通过 `conda run -n graph-mamba` 调用
    - 配置：`morphology-node-EX.yaml`（Mamba + GatedGCN + edge/node encoder + LapPE）
    - 示例调用：
      ```bash
      cd /home/kfchen/gnn_project
      bash scripts/run_graph_mamba_tiny50_full.sh
      ```
  - 在当前机器上运行时，由于 GPU 驱动过旧，会出现：
    - `CUDA initialization: The NVIDIA driver on your system is too old (found version 11040)`  
    - 最终在 Mamba CUDA kernel 处报 `Expected u.is_cuda() to be true, but got false.`
  - **在新机器（驱动兼容）上**，按相同环境重建后，此脚本应可直接在 GPU 上正常训练。

- **tiny50 两阶段 DiGress + Graph-Mamba 脚本**:  
  `scripts/run_two_stage_tiny50.sh`
  - 第一步：在 tiny50 上训练 DiGress 生成/去噪模型，产生 GraphTransformer encoder checkpoint。
  - 第二步：在 tiny50 上训练基于 DiGress encoder 的节点分类器。
  - 脚本内部目前仍使用硬编码的 `DATADIR_TINY50=/data2/.../synthesis_data_tiny_50`，后续可改为依赖 `GNN_DATA_ROOT`：
    - 建议接手者统一改为：
      ```bash
      DATADIR_TINY50="${DATADIR_TINY50:-$(python -c 'import config; print(config.TRAIN_DATA_DIR_TINY_50)')}"
      ```

### 4. 当前机器的限制与新机器建议

- **当前机器限制**:
  - NVIDIA 驱动版本仅支持 CUDA 11.4，无法满足 `torch 2.8.0+cu128` 的 GPU 要求。
  - 结果是：
    - Graph-Mamba EX（含 Mamba）目前在 CPU 上可跑，但 Mamba CUDA kernel 要求 GPU，导致训练时出错。
    - GatedGCN-only 配置（`--no-mamba`）不依赖 CUDA kernel，可在 CPU 上完全跑通。

- **新机器建议**:
  - 在新机器上只要：
    1. 驱动支持 CUDA 11.7+（最好 11.8 或 12.x）
    2. 按 `requirements_conda.txt` + `setup_graph_mamba_from_requirements.sh` 重建 `graph-mamba` 环境
    3. 设置好 `GNN_DATA_ROOT` 指向网络盘上的数据
  - 就可以直接：
    ```bash
    cd /home/<新机用户名>/gnn_project
    bash scripts/run_graph_mamba_tiny50_full.sh
    ```
    在 GPU 上训练最完整的 Graph-Mamba EX。

### 5. 两个新的 git worktree 提醒

本机上已额外创建了 **两个新的 git worktree**，用于并行开发/实验（例如一个用于 Graph-Mamba 适配，另一个用于 DiGress 流程或其它分支）：

- 接手者可在项目根目录执行：
  ```bash
  cd /home/kfchen/gnn_project
  git worktree list
  ```
  查看当前所有 worktree 的路径与对应分支。
- 建议：
  - 保留这两个 worktree（其中可能包含正在进行中的修改或实验分支）。
  - 在新机器上如果也需要同样的多分支开发模式，可按 `git worktree add <path> <branch>` 的方式在相同分支上重建 worktree。

### 6. 接手者建议的下一步操作

- **若你在新机器上继续**：
  1. 从仓库克隆代码到本地或网络盘：`/PBshare/SEU-ALLEN/Users/KaifengChen/gnn_project`。
  2. 设置 `GNN_DATA_ROOT` 指向网络盘数据根目录。
  3. 使用 `scripts/setup_graph_mamba_from_requirements.sh` 重建 `graph-mamba` 环境。
  4. 使用 `scripts/run_graph_mamba_tiny50_full.sh` 在 tiny50 上跑完整 EX 配置。

- **若暂时仍在本机继续**：
  1. 使用 `--no-mamba` 先在 CPU 上完成 GatedGCN-only 的 tiny50 实验：
     ```bash
     conda run -n graph-mamba python scripts/run_graph_mamba.py \
       --data_dir "$(python -c 'import config; print(config.TRAIN_DATA_DIR_TINY_50)')" \
       --wandb False \
       --name_tag tiny50_gatedgcn \
       --repeat 1 \
       --no-mamba
     ```
  2. 等待驱动升级或切换到新机器后，再启用完整 Mamba 配置。

### 7. “启发式扫描” worktree 思路与当前进展

> 该部分专门记录 `feature/heuristic-scan` 分支（worktree）上对 Graph-Mamba 扫描机制的改动，方便后续继续开发或论文撰写。

- **worktree 基本信息**
  - 分支：`feature/heuristic-scan`
  - worktree 路径：`/home/kfchen/gnn_project_heuristic_scan`
  - 创建方式（仅供新机器参考）：
    ```bash
    cd /home/<user>/gnn_project
    git worktree add -b feature/heuristic-scan ../gnn_project_heuristic_scan
    ```
  - 该分支独立于 `master`，主分支上的提交不会自动影响此 worktree，需在 worktree 内主动 `git merge master` 或 `git rebase master` 才会同步最新变更。

- **整体思路：让扫描顺序“懂结构”，而不是随机走**
  - Mamba 需要一个线性序列输入；在图上通常用 random walk / DFS / BFS 等将节点排成序列。
  - 在“噪声森林”场景（大量噪声点 + 一棵真实树）下，若序列一开始就访问到噪声节点，会污染 Mamba 的 hidden state，后面的“真树”节点读到的状态已经被破坏。
  - 本分支的目标：**让图上的扫描顺序由结构启发式 / GNN 得分来决定**，优先覆盖“像树”的骨干，把噪声推到序列后面，从而减轻状态污染。

- **已实现的启发式扫描（纯规则版）**
  - 修改文件：`external/Graph-Mamba/graphgps/layer/gps_layer.py`
  - 新增函数：
    - `tree_order_within_batch(edge_index, batch, order='bfs', root_choice='min_degree')`
      - 对 batch 中每个子图：
        - 构建局部邻接（按 `edge_index` 与 `batch` 划分）。
        - 根据 `root_choice` 选根（`min_degree` 叶子 / `max_degree` 类似 soma / `first` 第一节点）。
        - 以 BFS 或 DFS（`order='bfs'/'dfs'`）遍历得到节点访问顺序。
      - 返回：按每个图顺序拼接的全局下标排列，可直接用于 `to_dense_batch(h[perm], ...)`。
  - 在 `GPSLayer.forward` 中新增若干 `global_model_type` 分支：
    - `Mamba_TreeBFS`：BFS（层序），根 = 度最小的叶子节点。
    - `Mamba_TreeDFS`：DFS（先序），根 = 度最小的叶子节点。
    - `Mamba_TreeBFS_Soma`：BFS，根 = 度最大的节点（对神经元形态，近似 soma）。
  - 使用方式示例（Graph-Mamba YAML）：
    ```yaml
    gt:
      layer_type: CustomGatedGCN+Mamba_TreeBFS        # 或 Mamba_TreeDFS / Mamba_TreeBFS_Soma
      # 其余 gt.* 超参不变
    ```

- **已实现的 GNN-Guided Priority Serialization（GNN 引导的优先级扫描）**
  - 目标：不要只靠启发式规则（度/BFS），而是**让 GNN 先当“侦察兵”**：
    1. local GNN（如 `CustomGatedGCN`）计算节点 embedding `H_GNN = h_local`。
    2. 用一个小的 MLP 在 `H_GNN` 上预测“树似然得分” `s_i ∈ [0, 1]`。
    3. 以 `s_i` 为优先级做 **priority BFS**：始终优先扩展高 `s_i` 的节点，使骨干先被 Mamba 看到。
  - 代码改动要点（均在 `gps_layer.py`）：
    - 新增函数：
      - `gnn_priority_bfs_within_batch(edge_index, batch, scores)`
        - 每个图内：
          - 根：`argmax(scores)`（最像树的节点）。
          - 用堆实现“优先级 BFS”：frontier 里每次弹出分数最高的节点，依次扩展邻居。
        - 返回：按得分优先级 BFS 得到的全局下标排列。
    - 在 `GPSLayer.__init__` 中为 `global_model_type == 'Mamba_GNNPriorityBFS'` 注册：
      - 一个标准 Mamba 块：
        - `self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)`
      - 一个得分 MLP：
        - `self.gnn_priority_mlp = nn.Sequential(Linear(dim_h, dim_h) → ReLU → Dropout → Linear(dim_h, 1))`
    - 在 `GPSLayer.forward` 中新增分支：
      - 从 `h_out_list[0]` 取到本层的 local GNN 输出 `h_local`（若无 local 则退化为用 `h`）：
        - `h_local = h_out_list[0] if h_out_list else h`
      - 计算树似然得分：
        - `scores_i = sigmoid(self.gnn_priority_mlp(h_local).squeeze(-1))`
      - 调用优先级 BFS 得到排列：
        - `h_ind_perm = gnn_priority_bfs_within_batch(batch.edge_index, batch.batch, scores_i)`
      - 按该顺序打包序列并送入 Mamba，再按 `h_ind_perm_reverse` 还原顺序：
        - 与其他 `Mamba_*` 分支保持同样的 residual / norm 逻辑。
  - 对应配置示例：
    ```yaml
    gt:
      layer_type: CustomGatedGCN+Mamba_GNNPriorityBFS
    gnn:
      layer_type: gatedgcnconv   # 作为 local GNN，为 MLP 提供 H_GNN
    ```

- **当前在 tiny50 上的 debug 进展（medsam 环境）**
  - 由于 `conda medsam` 环境中：
    - **有**：`torch`、`torch_geometric`，可以跑 GraphGym/Graph-Mamba；
    - **无**：`mamba_ssm`，暂时不能在该环境下启用 Mamba 分支；
    - **无**：DiGress 所需 `hydra`，两阶段脚本在 medsam 下无法直接运行。
  - 已在 medsam 中验证 **tiny50 + GatedGCN-only** 流程可跑通，用于数据与训练管线 debug：
    ```bash
    cd /home/kfchen/gnn_project
    conda run -n medsam python scripts/run_graph_mamba.py \
      --data_dir /data2/kfchen/tracing_ws/morphology_seg/synthesis_data_tiny_50 \
      --no-mamba \
      --override optim.max_epoch 3 \
      --override train.batch_size 4
    ```
  - 该命令使用 `morphology-node-GatedGCN-only.yaml`，主要用来：
    - 验证 MorphologyNodeDataset + GraphGym pipeline 在 tiny50 上无路径错误；
    - 粗看数值稳定性（当前 run 中 loss 出现 `nan`，后续可在 `custom_train` 里对 loss/logits 做 clamp 或检查标签分布）。

- **后续建议（针对“启发式扫描”分支）**
  1. 在 **支持 mamba-ssm 的环境**（如 `graph-mamba` 或新机器）中：
     - 用 `Mamba_TreeBFS` / `Mamba_TreeBFS_Soma` / `Mamba_GNNPriorityBFS` 在 tiny50/full 数据上系统对比：
       - 骨干覆盖情况（可视化扫描顺序）；
       - 对噪声注入程度不同的数据集的鲁棒性。
  2. 若要写论文：
     - Part 1：以 `Mamba_Tree*` + `Mamba_GNNPriorityBFS` 为主线，讲“Structure-Aware Scan”；
     - Part 2（可选）：在 mamba-ssm 内进一步让 Δ 受 `s_i` 调制，做真正的“Selective Scan”（这一部分目前只在思路层面，未在本仓库实现）。

