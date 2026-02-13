# mamba-ssm 安装失败原因与解决

**若仍有问题**，可尽量按 **`external/Graph-Mamba/requirements_conda.txt`** 重新配置环境（Python 3.9、PyTorch 2.0.0 cu117、MKL<2024.1、mamba-ssm 等），一键脚本：

```bash
cd /path/to/gnn_project
bash scripts/setup_graph_mamba_from_requirements.sh graph-mamba
```

详见下方「按 requirements_conda.txt 配置」小节。

---

## graph-mamba 中 PyTorch 2.8+cu128 与 nvcc 11.7 不匹配（当前常见情况）

若环境中 **PyTorch 已是 2.8.0+cu128**（被 torch-geometric 等依赖升级），而 **nvcc 仍是 11.7**，会出现：

```text
RuntimeError: The detected CUDA version (11.7) mismatches the version that was used to compile PyTorch (12.8).
```

且 mamba-ssm 2.3.0 会优先尝试从 GitHub 下载 **cu12torch2.8** 的预编译 wheel，该文件在 release 中不存在或下载易失败（如 “Remote end closed connection”）。

### 解决步骤（任选其一）

**方案 A：让 nvcc 与 PyTorch 一致后从源码构建（推荐）**

1. 在 graph-mamba 中安装与 PyTorch CUDA 一致的 nvcc（如 12.8）：
   ```bash
   conda activate graph-mamba
   conda install -y nvidia/label/cuda-12.8.0::cuda-nvcc
   nvcc -V   # 应显示 12.x
   ```
2. 强制从源码构建并安装：
   ```bash
   pip install ninja packaging -q
   MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm==2.3.0 --no-build-isolation
   ```
3. 验证：`python -c "import mamba_ssm; print('OK')"`

**方案 B：网络稳定时用预编译 wheel（torch2.7 与 2.8 通常兼容）**

1. 下载 cp39、cu12、torch2.7、cxx11abiTRUE 的 wheel（约 509MB）：
   ```text
   https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.7cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
   ```
2. 下载完成后安装（使用**完整文件名**，不要改名为简化名）：
   ```bash
   pip install --no-deps /path/to/mamba_ssm-2.3.0+cu12torch2.7cxx11abiTRUE-cp39-cp39-linux_x86_64.whl
   ```
3. 若之前下载不完整（如 458MB），wheel 无效，需重新完整下载后再装。

**方案 C：不装 mamba-ssm，仅用 GatedGCN**

- 运行：`python scripts/run_graph_mamba.py --no-mamba --wandb False`  
- 见 `docs/GRAPH_MAMBA_RUN.md`。

一键脚本（先装匹配的 nvcc，再强制源码构建）：见 **`scripts/install_mamba_ssm_graph_mamba.sh`**。

若安装后 `import mamba_ssm` 报错 **`libopenblas.so.0: cannot open shared object file`**，可安装 OpenBLAS 并保证环境 lib 被加载：

```bash
conda activate graph-mamba
conda install -y libopenblas -c conda-forge
# 运行时确保环境 lib 在 LD_LIBRARY_PATH 中（激活环境后通常已包含）
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
python -c "import mamba_ssm; print('OK')"
```

---

## 失败原因

安装时报错：

```text
RuntimeError: mamba_ssm is only supported on CUDA 11.6 and above.
Note: make sure nvcc has a supported version by running nvcc -V.
```

- **mamba-ssm** 在构建时会检查 **系统/环境里的 `nvcc -V`**，要求 **CUDA ≥ 11.6**。
- 当前机器上 `nvcc` 来自 `/usr/bin/nvcc`，版本为 **CUDA 10.1**，不满足要求。
- 因此即使用 conda 里 PyTorch 是 cu118，只要 `nvcc -V` 仍是 10.1，安装就会失败。

另外，若安装一直停在 “Installing build dependencies” 或编译阶段，多半是：
- 在等构建依赖下载/编译，或
- 之前已经因 nvcc 版本被拒绝，但错误被缓冲/未完全显示。

## 解决办法

### 方案一：在当前 conda 环境中安装 CUDA 11.x 的 nvcc（推荐）

在**要用来装 mamba-ssm 的环境**（例如 `digress`）里安装 NVIDIA 官方提供的 nvcc，使该环境内的 `nvcc -V` ≥ 11.6：

```bash
conda activate digress   # 或你的 PyTorch+cu118 环境
conda install -c nvidia cuda-nvcc=11.8
nvcc -V   # 应显示 11.8
pip install mamba-ssm
```

若希望与 PyTorch 的 CUDA 一致，可装 11.8；若系统/驱动支持，也可装 12.x（如 `cuda-nvcc=12.1`）。

### 方案二：系统安装 CUDA 11.6+ 并优先使用其 nvcc

在机器上安装 CUDA Toolkit 11.6 或 11.8（例如到 `/usr/local/cuda-11.8`），安装后在**当前 shell** 里把该版本放在 PATH 最前，再安装：

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
nvcc -V   # 确认 ≥ 11.6
pip install mamba-ssm
```

需要 root 或已有 CUDA 11.x 安装目录。

### 方案三：不装 mamba-ssm，仅用 GatedGCN

若暂时无法升级 nvcc，可继续使用**不需要 mamba-ssm** 的配置：

- 配置：`configs/Mamba/morphology-node-GatedGCN-only.yaml`
- 运行：`python scripts/run_graph_mamba.py --no-mamba --wandb False`

详见 `docs/GRAPH_MAMBA_RUN.md`。

## 小结

| 现象 | 原因 |
|------|------|
| `RuntimeError: mamba_ssm is only supported on CUDA 11.6 and above` | 当前 `nvcc -V` 是 10.1，不满足 mamba-ssm 要求 |
| 安装卡在 “Installing build dependencies” 或编译 | 可能之前因 nvcc 检查失败，或编译/网络较慢 |

**根本解决**：让构建时用到的 `nvcc` 版本 ≥ 11.6（推荐用 conda 在目标环境中装 `cuda-nvcc=11.8`）。

---

## 安装成功但 import 报错：GLIBC 版本

在 **graph-mamba** 环境中已用以下方式成功安装 mamba-ssm 2.3.0：

```bash
conda activate graph-mamba
pip install ninja packaging -q
pip install mamba-ssm --no-build-isolation
```

但运行 `import mamba_ssm` 时报错：

```text
ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
```

- **原因**：本地编译出的 CUDA 扩展依赖 **GLIBC ≥ 2.32**，当前系统为 **Ubuntu GLIBC 2.31**。
- **可行做法**：
  1. **在 glibc ≥ 2.32 的环境跑**：如 Ubuntu 22.04+ 或 Docker 容器（例如 `ubuntu:22.04`），在同一环境中用 `--no-build-isolation` 安装并运行。
  2. **本机继续用 GatedGCN**：不 `import mamba_ssm`，使用 `python scripts/run_graph_mamba.py --no-mamba`，见 `docs/GRAPH_MAMBA_RUN.md`。

---

## 在 Ubuntu 20.04（glibc 2.31）上能用吗？

**可以**，不是“永远无法”，只是要避免本机用 conda 的 lib 链接出依赖 GLIBC_2.32 的 .so。可选做法：

### 做法一：Docker 里用 Ubuntu 20.04 构建并运行（推荐）

在**同为 Ubuntu 20.04** 的 Docker 容器里安装 PyTorch（cu118）+ mamba-ssm，编译出的 .so 只依赖 glibc 2.31，本机也是 20.04 时可直接用。

```bash
# 宿主机
docker run -it --gpus all -v /path/to/gnn_project:/workspace ubuntu:20.04 bash

# 容器内：装 miniconda、Python 3.10、CUDA 11.8 的 PyTorch、nvcc 11.8、再装 mamba-ssm
# 全部在 20.04 内完成，得到的 mamba_ssm 即可 import
```

之后在容器里跑完整 Graph-Mamba，或把容器内 site-packages 里的 `mamba_ssm` 拷到本机同版本 Python 环境里用也可（本机 glibc 仍是 2.31 即可）。

### 做法二：本机从源码装，强制用系统 lib 链接（可能可行）

本机用 pip 从源码装时，让**链接阶段**只用系统 glibc/libstdc++，避免 conda 的 lib 被链进去（否则容易带上 2.32 依赖）：

```bash
conda activate graph-mamba
# 临时不用 conda 的 lib，只用系统路径（按你实际路径调整）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
pip install ninja packaging -q
pip install mamba-ssm --no-build-isolation
```

若仍出现 GLIBC_2.32，说明编译/链接仍用到了带 2.32 的库，可改用**系统 Python + venv**（不用 conda）在干净环境里装 PyTorch + mamba-ssm，再试。

### 做法三：conda-forge 预编译包（可能换掉 PyTorch/CUDA）

```bash
conda install -c conda-forge mamba-ssm
```

conda-forge 的 mamba-ssm 可能在较老 glibc 上能 import，但会拉取 conda-forge 的 PyTorch 等依赖，**可能变成 CPU 版或其它 CUDA 版本**，若你要固定用当前 PyTorch cu118，需谨慎或单独建一个环境再试。

---

小结：**Ubuntu 20.04 下可以安装并运行 mamba-ssm**，最稳的是在 **Ubuntu 20.04 的 Docker 里** 用该系统的 glibc 构建；本机若坚持用 conda，可尝试做法二或三。

---

## 按 requirements_conda.txt 配置（若仍有问题）

尽量对齐 **`external/Graph-Mamba/requirements_conda.txt`** 时，主要版本为：

| 项目 | 版本 |
|------|------|
| Python | 3.9 |
| PyTorch | 2.0.0 (CUDA 11.7) |
| torch-geometric | 2.0.4 |
| torchvision / torchaudio | 0.15.0 / 2.0.0 (cu117) |
| MKL | &lt;2024.1（避免 iJIT_NotifyEvent） |
| cuda-nvcc | 11.7 |
| mamba-ssm | 1.0.1（或 2.3.0） |

**一键脚本**（会删除并重建指定环境，默认 `graph-mamba`）：

```bash
cd /path/to/gnn_project
bash scripts/setup_graph_mamba_from_requirements.sh [环境名]
# 示例：bash scripts/setup_graph_mamba_from_requirements.sh graph-mamba
```

脚本会：创建 Python 3.9 环境 → 安装 PyTorch 2.0.0 cu117 → 限制 MKL&lt;2024.1 → 安装 cuda-nvcc 11.7 → pip 安装 PyG、ogb、yacs、tensorboardx、performer-pytorch、torchmetrics 及 PyG 扩展（pt20 cu117）→ 以 `--no-build-isolation` 安装 mamba-ssm（先试 1.0.1，失败则试 2.3.0）。若本机驱动/库仅为 cu118，可自行把脚本中的 11.7 改为 11.8 并改用对应 PyG wheel 索引。
