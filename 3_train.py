import os
import glob
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool

# ==========================================
# 1. 特征预处理逻辑 (复用之前的逻辑)
# ==========================================
def preprocess_features(data):
    """
    实时处理单个图的数据：One-Hot + 归一化
    """
    x = data.x

    # --- A. 分离特征 ---
    # 假设原始 input: [Radius, Type, Dist, Angle]
    radius = x[:, 0:1]
    node_type = x[:, 1].long()
    dist = x[:, 2:3]
    angle = x[:, 3:4]

    # --- B. One-Hot ---
    type_one_hot = F.one_hot(node_type, num_classes=3).float()

    # --- C. 归一化 (Per Graph Standardization) ---
    # 注意：这里是在每个单独的树内部做归一化
    def standardize(tensor):
        mask = tensor >= 0
        if mask.sum() > 1: # 至少有2个点才能算方差
            mean = tensor[mask].mean()
            std = tensor[mask].std() + 1e-6
            return (tensor - mean) / std
        return tensor

    radius_norm = standardize(radius)
    dist_norm = standardize(dist)
    angle_norm = standardize(angle)

    # --- D. 拼接 ---
    # 新维度: 1 + 3 + 1 + 1 = 6
    new_x = torch.cat([radius_norm, type_one_hot, dist_norm, angle_norm], dim=1)

    data.x = new_x
    data.y = data.y.long()
    return data

# ==========================================
# 2. 自定义数据集加载器 (核心修改)
# ==========================================
class TreeDataset(Dataset):
    def __init__(self, root_dir):
        # 初始化基类
        super().__init__(root_dir, transform=None, pre_transform=None)
        self.root_dir = root_dir

        # === 修复点 ===
        # 原来的代码使用了 'self.processed_file_names'，这是 PyG 的保留只读属性，会导致报错。
        # 我们将其改名为 'self.pt_files' (或者任何其他名字)。
        self.pt_files = glob.glob(os.path.join(root_dir, "*.pt"))

        # 简单的排序，保证每次加载顺序一致
        self.pt_files.sort()

        print(f"Found {len(self.pt_files)} graph files.")

    def len(self):
        # === 修复点 ===
        # 读取我们自定义的列表长度
        return len(self.pt_files)

    def get(self, idx):
        # === 修复点 ===
        # 从我们自定义的列表中获取路径
        data_path = self.pt_files[idx]

        # 加载数据
        data = torch.load(data_path)

        # 实时预处理 (One-Hot + Norm)
        data = preprocess_features(data)

        return data

# ==========================================
# 3. 定义模型 (GCN)
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x) # 节点嵌入

        # Classifier
        # 直接对每个节点进行分类
        out = self.classifier(x)

        return out

# ==========================================
# 4. 训练流程
# ==========================================

# --- 配置路径 ---
DATA_FOLDER = f"/data2/kfchen/tracing_ws/morphology_seg/synthesis_data" # 你的文件夹路径，里面放着很多 .pt 文件

# --- 加载数据 ---
dataset = TreeDataset(DATA_FOLDER)

# --- 划分训练集和测试集 (按图划分) ---
# 假设 80% 的图用于训练，20% 的图用于测试
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# --- 创建 DataLoader (Batching) ---
# batch_size=4 表示一次把 4 个树拼成一个大图扔进 GPU
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# --- 初始化模型 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=6, hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# --- 训练函数 ---
def train():
    model.train()
    total_loss = 0
    # 遍历 Batch
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch) # out shape: [Batch_Total_Nodes, 2]

        # 计算 Loss
        # 注意：这里我们使用 batch 中“所有”节点的 label 进行训练
        # 因为这是 Inductive Learning，我们已经把测试集的图完全隔离开了
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# --- 测试函数 ---
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1)

        correct += (pred == batch.y).sum().item()
        total_nodes += batch.num_nodes

    return correct / total_nodes

model_save_path = "/data2/kfchen/tracing_ws/morphology_seg/best_model.pth"
best_test_acc = 0
for epoch in range(51):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    # === 新增：保存最佳模型的逻辑 ===
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        # 保存模型参数 (State Dict)
        torch.save(model.state_dict(), model_save_path)
        print(f'Epoch: {epoch:03d} -> New Best Model Saved! (Test Acc: {test_acc:.4f})')
    # ==============================

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print(f"Training finished. Best model saved to {model_save_path} with acc {best_test_acc:.4f}")