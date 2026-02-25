# 训练中出现 NaN 的原因与应对

## 现象（本次 BFS baseline）

- 约 **epoch 52–56**：`val_loss` 出现 `nan`，`val_accuracy` 仍正常（0.8974–0.9005）。
- 约 **epoch 57** 及之后数轮：`test_loss` 为 `nan`，`test_accuracy` 正常。
- 训练里已有保护：`custom_train.py` 中若 `loss` 为 NaN/Inf 会**跳过 backward 和 step**，只做 `optimizer.zero_grad()`，因此训练没有立刻崩掉；但 **eval 时仍会计算并记录该 batch 的 loss**，所以日志里会看到 `val_loss: nan` / `test_loss: nan`。

---

## 根本原因（链条）

1. **记录到的 loss 来自**：`compute_loss(pred, true)`（形态学节点任务走默认的 cross_entropy：`log_softmax(pred) + nll_loss`）。
2. **loss 变成 NaN 只可能有两种情况**：
   - **pred（logits）里已有 nan/inf**：前向某处产生 nan/inf，一路传到 loss。
   - **pred 数值极大**：float32 下 `exp(logit)` 在 logit > 88 左右会溢出为 inf，`log_softmax` 会得到 -inf/nan，进而 `nll_loss` 得到 nan。

因此，**NaN 的本质是：在部分 val/test batch 上，模型前向得到的 logits 要么本身是 nan/inf，要么过大导致 log_softmax 溢出。**

---

## 可能来源（按优先级）

### 1. Mamba SSM 内部数值（最可疑）

- Mamba 里对 `dt` 做 `exp(dt)`、以及 selective scan 的递推，若 `dt` 或其它输入变大，容易 **exp 溢出 → inf → 后续运算产生 nan**。
- 训练时某些 batch 的激活或梯度会让 SSM 的中间量暂时变大，在 **eval 时**（不同 batch、不同图结构）也可能在个别 batch 上触发同样问题，表现为 **仅 val/test 的 loss 为 nan**，与当前日志一致。

### 2. BatchNorm 在 eval 下的方差为 0

- 在 `model.eval()` 下用 **running_var**。若某通道在训练中几乎不变，`running_var` 可能非常小或为 0，做 `x / sqrt(running_var + eps)` 时若 `eps` 过小或数值不当，可能得到 inf/nan。
- 若验证/测试集某个 batch 的节点/图分布与训练差异大，也可能放大这类问题。

### 3. 辅助 loss（GNN priority BCE）已做保护，不是主因

- `_add_gnn_priority_aux_loss` 里已对 `scores` 做 `clamp(1e-6, 1-1e-6)` 和 `nan_to_num`，且主 loss 的 nan 来自 **主任务 logits**，不是来自辅助 BCE。

### 4. 与 fusion 门控无关

- 当前 BFS baseline 使用 `gt.fusion = 'sum'`，**未启用** `ConflictAwareFusion`，因此本次 NaN 与门控或 ReLU 无关。

---

## 建议应对

1. **保持现有梯度与 NaN 保护**  
   继续保留：`if not (torch.isnan(loss).any() or torch.isinf(loss).any()): loss.backward() ...` 以及 `clip_grad_norm`，避免 nan 梯度写回参数。

2. **对主任务 logits 做数值保护（已实现）**  
   在 `custom_train.py` 中已添加 `_safe_logits(pred)`：在 **train_epoch** 与 **eval_epoch** 中，在得到 `pred, true = model(batch)` 后、计算 loss 前，执行 `pred = _safe_logits(pred)`。内部逻辑：`pred = torch.nan_to_num(pred, nan=0.0, posinf=50.0, neginf=-50.0)` 再 `pred = torch.clamp(pred, min=-50.0, max=50.0)`，避免 log_softmax 溢出与 loss 为 nan。

3. **评估时遇到 nan 的 fallback**  
   在 eval 的 logger 里，若当前 batch 的 loss 为 nan/inf，可以不累加该 batch 的 loss，或用上一轮有效 loss 代替，避免整轮 val/test loss 显示为 nan（可选，主要用于日志可读性）。

4. **Mamba-SSM 的数值稳定写法（本项目已采用）**  
   **mamba-ssm** 官方的数值稳定设计在 **初始化** 阶段（见 [mamba_simple.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py)）：
   - **dt_min=0.001, dt_max=0.1**：离散化步长 dt 的允许范围；初始化时令 `F.softplus(dt_bias)` 落在此区间，避免 dt 过小/过大导致 exp 溢出或递推不稳定。
   - **dt_init_floor=1e-4**：dt 的下界地板。
   - **delta_softplus=True**：前向时对 dt 使用 softplus，保证为正；但训练后 `dt_proj` 输出可能变大，softplus(dt) 仍可能超出 [dt_min, dt_max]，故仅靠 init 不能完全杜绝溢出。
   - 本项目在 `gps_layer.py` 中创建 `Mamba(...)` 时已**显式传入** `dt_min=0.001, dt_max=0.1`（与官方默认一致），保证使用官方推荐的稳定区间；若仍出现 NaN，可尝试将 `dt_max` 调小（如 0.05）再训练。
   - 若需从根上限制**前向时**的 dt：需在 mamba-ssm 的 forward 或 selective_scan 内核内对 dt 做 clamp（官方 Mamba 1 未在每步 clamp，Mamba 2 有 `dt_limit` 等扩展）。当前更稳妥的做法仍是 **loss 前对 pred 做 `_safe_logits`**（已实现）。

5. **BatchNorm**  
   确认所有 BN 的 `eps >= 1e-5`，必要时略调大（如 1e-4），减少 div-by-zero 风险。

---

## 小结

| 项目         | 说明 |
|--------------|------|
| **直接原因** | 部分 val/test batch 上，主任务 `pred` 为 nan/inf 或过大，导致 `log_softmax` + `nll_loss` 得到 nan。 |
| **最可能来源** | Mamba SSM 中 `exp(dt)` 或递推导致的数值爆炸；其次为 BN 在 eval 下的极端方差。 |
| **已存在保护** | 训练时 nan loss 不 backward、不 step，并做了梯度裁剪。 |
| **建议** | 对 `pred` 做 clamp/nan_to_num 再算 loss（**已实现**：`_safe_logits`）；必要时在 Mamba 内对 dt/exp 做数值稳定处理。 |
