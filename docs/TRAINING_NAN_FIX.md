# 训练 Loss=NaN 原因与处理

## 原因说明

1. **原始 .pt 中 path distance 含 Inf**  
   `compute_geodesic_distance` 对从 root 不可达的节点会返回 `inf`。这些 Inf 写入 `x` 的 dist 列，经标准化后仍会得到 Inf/NaN，前向与 loss 会变成 NaN。

2. **Graph-Mamba 使用了旧的 processed 缓存**  
   即使已用 `scripts/fix_pt_inf.py` 修复了 .pt 文件，若数据目录下已有 `processed/`（如 `synthesis_data/processed/`），MorphologyNode 会直接加载该缓存而**不会**重新从 .pt 读入。该缓存若在修复前生成，其中特征仍含 Inf，训练会从第一个 epoch 起就 loss=nan。

## 处理步骤

1. **修复 .pt 中的 Inf/NaN**（若尚未做）  
   ```bash
   python scripts/fix_pt_inf.py --data_dir /data2/kfchen/tracing_ws/morphology_seg/synthesis_data
   ```

2. **删除 processed 缓存，强制用修复后的 .pt 重新处理**  
   ```bash
   rm -rf /data2/kfchen/tracing_ws/morphology_seg/synthesis_data/processed
   ```

3. **重新启动训练**  
   下次运行 `python scripts/run_graph_mamba.py --no-mamba --wandb False` 时，会从 .pt 重新生成 processed，特征中不再含 Inf，loss 应恢复正常。

## 代码侧已做防护

- **数据生成**（`data/synthesis.py`）：在 `convert_to_gnn_data` 中已将非有限的 dist/angle 置为 -1/0，新生成的 .pt 不会带 Inf。
- **训练**（`graphgps/train/custom_train.py`）：若某步 loss 为 NaN/Inf，会跳过该步的 backward 与 optimizer.step，避免梯度污染导致后续全为 NaN。
