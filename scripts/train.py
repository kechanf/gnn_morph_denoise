"""
入口脚本：使用配置中的数据与超参数训练 GCN，并保存最佳模型。
用法: 在项目根目录执行  python scripts/train.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import DataLoader

import config
from data.dataset import TreeDataset
from models.gcn import GCN


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total_nodes += batch.num_nodes
    return correct / total_nodes if total_nodes else 0.0


def main():
    dataset = TreeDataset(config.TRAIN_DATA_DIR)
    train_size = int(config.TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(
        config.IN_CHANNELS,
        config.HIDDEN_CHANNELS,
        config.OUT_CHANNELS,
        dropout=config.DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH) or ".", exist_ok=True)
    best_metric = 0.0

    for epoch in range(config.NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        if config.SAVE_BEST_BASED_ON == "test":
            current = test_acc
        else:
            current = train_acc
        if current > best_metric:
            best_metric = current
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Epoch {epoch:03d} -> Best model saved (acc: {current:.4f})")

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}"
            )

    print(f"Training finished. Best model: {config.MODEL_SAVE_PATH} (acc: {best_metric:.4f})")


if __name__ == "__main__":
    main()
