# 数据合成策略说明

本项目支持多种数据合成策略，用于模拟不同类型的神经元追踪噪声/错误。

## 策略列表

### 1. `full_tree`（整棵干扰树）
**描述**：将整棵干扰树随机旋转后，随机贴到目标树的某个节点上。  
**特点**：
- 模拟“多棵树合并”的场景（如自动追踪将相邻神经元误合并）
- 干扰规模大（整棵树）
- 干扰节点数量多

**参数**：
- `interfer_paths`: 干扰 SWC 文件路径列表

**示例**：
```python
generate_dataset(target_path, interfer_paths, out_path, strategy="full_tree")
```

---

### 2. `local_spur`（局部短刺）
**描述**：在目标树节点上生成短刺状假分支（3-8 个节点，线性路径）。  
**特点**：
- 模拟“追踪错误导致的短假分支”
- 干扰规模小（局部）
- 更贴近真实追踪错误（如成像伪影导致的短刺）

**参数**：
- `length`: 假分支长度（默认 3-8 随机）
- `max_radius`: 最大半径（默认 2.0）

**示例**：
```python
generate_dataset(target_path, [], out_path, strategy="local_spur")
# 注意：local_spur 不需要 interfer_paths
```

---

### 3. `branch_segment`（分支片段）
**描述**：从干扰树中提取一段路径（3-10 个节点），贴到目标树上。  
**特点**：
- 介于“整棵树”和“短刺”之间
- 保留干扰树的局部形态特征
- 干扰规模中等

**参数**：
- `interfer_paths`: 干扰 SWC 文件路径列表
- `segment_length`: 路径长度（默认 3-10 随机）

**示例**：
```python
generate_dataset(target_path, interfer_paths, out_path, strategy="branch_segment")
```

---

### 4. `small_cluster`（小点簇）
**描述**：在目标树节点附近生成小的随机点簇（2-5 个节点，用 MST 连接）。  
**特点**：
- 模拟“成像伪影/噪声点”
- 干扰规模很小
- 点簇结构简单（MST 连接）

**参数**：
- `num_points`: 点簇节点数（默认 2-5 随机）
- `radius_um`: 点簇半径（默认 3.0-8.0 随机）

**示例**：
```python
generate_dataset(target_path, [], out_path, strategy="small_cluster")
# 注意：small_cluster 不需要 interfer_paths
```

---

### 5. `mixed`（混合策略）
**描述**：随机混合使用上述所有策略。  
**特点**：
- 每个干扰源随机选择一种策略
- 数据多样性最高
- 更接近真实场景的复杂性

**示例**：
```python
generate_dataset(target_path, interfer_paths, out_path, strategy="mixed")
```

---

## 使用示例

### 生成单个样本（用于观察）

```bash
python scripts/generate_samples.py
```

这会为每种策略生成一个样本，保存在 `{DATA_ROOT}/sample_visualization/` 目录下：
- `sample_{strategy}_full.swc` - 完整合并图
- `sample_{strategy}_target_only.swc` - 仅目标节点
- `sample_{strategy}_interfer_only.swc` - 仅干扰节点

### 批量生成训练数据

修改 `scripts/synthesis_data.py`，在调用 `generate_dataset` 时指定 `strategy` 参数：

```python
# 使用混合策略
generate_dataset(target_path, interfer_paths, out_path, 
                dist_threshold=config.MERGE_DIST_THRESHOLD,
                strategy="mixed")
```

---

## 策略对比

| 策略 | 干扰规模 | 干扰节点数 | 真实度 | 适用场景 |
|------|---------|-----------|--------|---------|
| `full_tree` | 大 | 多（整棵树） | 中 | 多树合并错误 |
| `local_spur` | 小 | 少（3-8） | 高 | 追踪假分支 |
| `branch_segment` | 中 | 中（3-10） | 中高 | 局部形态错误 |
| `small_cluster` | 很小 | 很少（2-5） | 高 | 成像伪影 |
| `mixed` | 混合 | 混合 | 最高 | 综合场景 |

---

## 观察生成的样本

生成的 SWC 文件可用以下工具查看：
- **Vaa3D**: 推荐，支持 3D 可视化
- **neuTube**: 开源神经元追踪工具
- **其他 SWC 查看器**

建议对比观察：
1. `*_full.swc` - 查看合成后的完整图
2. `*_target_only.swc` - 查看目标树（ground truth）
3. `*_interfer_only.swc` - 查看干扰部分

这样可以直观理解每种策略产生的噪声模式。
