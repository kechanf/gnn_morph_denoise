# 工作日志 (Work Log)

## 2026-02-09

- **数据合成策略**：将原有几种噪声归为「简单负样本」，并新增「断裂+碎片」高级策略；碎片改为从 seed 库真实 SWC 中截取，虚假连接改为基于空间近邻（目标节点/断裂端点），并增加碎片与断裂端点的错误连接。
- **角度约束**：对新连接（树连接、分支连接、碎片连接、小簇连接）增加几何约束，使新分支与参考方向夹角在配置阈值内随机（如 30°），而非对齐为 0°，更不易被分辨。
- **配置与参数**：数据合成与训练参数分离到 `config.py` 并加注释；各策略参数（如 `BREAK_FRAGMENT_CFG`、`SIMPLE_NEG_STRATEGY_WEIGHTS`）可调；新增 `SYNTHESIS_SAMPLE_TIMEOUT_SEC` 单样本超时。
- **合成脚本**：`scripts/synthesis_data.py` 增加单样本超时判定（子进程 + 超时则跳过并删除不完整 .pt），避免卡在某一样本。
- **训练与数据**：清空旧 .pt 后仅用新生成的 2000 个 mixed 策略样本训练；自动训练调度在样本数达到 2000 后启动（当前合成曾卡在 1529，已停掉并改为带超时的合成逻辑，暂未重新生成）。
- **Git**：配置 GitHub 仓库（kechanf）；添加 `.gitignore`（Python/IDE/日志/生成数据）；SSH 推送需配置公钥到 GitHub。

## 2026-02-10

- **DiGress 基线部署**：在 `external/DiGress` 成功引入 DiGress 代码，完成环境安装与 `orca` 编译，并在本项目中写好一键脚本 `scripts/install_digress.sh`（含 `--solver classic` 与清华源配置说明，避免 `libmamba` 报错）。
- **Morphology 抽象图数据集**：实现 `morphology_dataset.py`（`MorphologyGraphDataset/DataModule/DatasetInfos`），将形态学图转换为 DiGress 的抽象图表示（节点全 1、边为 [no_edge, edge] one‑hot），并新增 `configs/dataset/morphology.yaml`；在 tiny50 子集和完整 `synthesis_data` 上均成功跑通无条件 DiGress（batch_size 调整为 1 以避免 OOM）。
- **训练指标扩展与日志体系**：在 `abstract_metrics.py` 中扩展 `TrainAbstractMetricsDiscrete`，记录 token 级 `node_acc`、`edge_acc`、`edge_precision`、`edge_recall`；在 `diffusion_model_discrete.py` 中增加 `log_text`，所有关键指标与采样信息自动写入 `logs/digress_*.log`，并修复 `wandb`、`CSVLogger` 超参数保存导致的异常。
- **成对去噪数据集 (MorphologyDenoise)**：设计并实现 `morphology_denoise_dataset.py`，从原始图构造 `(G_noise, G_target)` 成对样本：`G_noise` 为完整噪声森林（记录在 `x_noise/edge_index_noise/edge_attr_noise` 等字段），`G_target` 为 `orig.y==1` 节点诱导的干净目标树，并提供对应的 `DataModule/DatasetInfos` 与 `configs/dataset/morphology_denoise.yaml`。
- **条件 DiGress（噪声森林 → 目标树）原型**：新增 `NoiseGraphEncoder`，在 `DiscreteDenoisingDiffusion` 中接入条件向量 `cond_y`（由 `G_noise` 编码得到），通过 `y` 通道对生成过程进行条件约束；在 tiny50 上验证条件去噪训练/验证/测试全流程可跑通。
- **稳定性修复**：定位并修复多个关键问题：包括 `torch.load(weights_only=False)` 解决自定义 PyG `Data` 反序列化错误、`max_nodes_possible` 提升到 5000 避免节点计数越界、为 `sampling_metrics` 增加 None 判空保护、将 `kl_prior` 中 `Ts = self.T` 改为 `Ts = (self.T - 1)` 以避免 CUDA device-side assert 越界等，确保 tiny50 和 full 数据集上训练/测试过程稳定。

## 2026-02-12

- **Graph-Mamba 深度扫描（第三阶段）**：围绕固定配置（`dim_inner=192, dropout=0.1, lr=0.003, bs=16`）完成 `layers=7..20` 的系统实验。结果显示从 7 到 20 层整体持续提升并在高层趋于平台，`val_acc` 最优在 `layers=19`（0.86906），`val_auc` 最优在 `layers=20`（0.94967）。
- **20 层新 baseline 确立**：将 20 层作为新 baseline，并在 `scripts/run_graph_mamba_experiments.py` 中新增第四阶段 `s4_*` 调参预设（学习率、dropout、隐藏维度、weight decay、batch size）。
- **第四阶段调参结果**：完成 `s4` 全部 10 组实验并输出 `graph_mamba_experiments_stage4.csv`。单项最优结果为：`s4_wd_0_02` 取得最高 `val_acc=0.87154`，`s4_drop_0_15` 取得最高 `val_auc=0.94991`。
- **组合验证实验编排（第五阶段）**：基于第四阶段最优方向新增 `s5_combo_*` 四组组合预设（`wd=0.02`、`dropout=0.15`、`lr=0.0025` 的两两/三项组合），并启动批量实验，结果文件目标为 `graph_mamba_experiments_stage5_combo.csv`。
- **实验脚本持续维护**：`scripts/run_graph_mamba_experiments.py` 已按阶段化方式扩展（`s3` 深度扫描、`s4` 单项调参、`s5` 组合验证），支持通过 `--presets` 精确选择实验组并统一写入指定 CSV。
