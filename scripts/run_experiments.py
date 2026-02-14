"""
按 config.EXPERIMENT_PRESETS 批量跑对照实验：每个 preset 训练一个模型，
保存到 DATA_ROOT/checkpoints/{run_name}.pth，并汇总结果到 CSV。

用法: 在项目根目录执行  python scripts/run_experiments.py
可选: 只跑部分预设，例如  python scripts/run_experiments.py --presets 0 1 2
"""
import sys
import os
import csv
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import DataLoader

import config
from data.dataset import TreeDataset
from models.build import build_model


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


def run_one_experiment(preset: dict, train_loader, test_loader, device, checkpoints_dir):
    """
    跑单个预设实验，返回结果字典（用于写 CSV）。
    """
    run_name = preset["run_name"]
    model_type = preset["model_type"]
    num_layers = preset["num_layers"]
    hidden_channels = preset["hidden_channels"]
    dropout = preset["dropout"]
    lr = preset.get("learning_rate", config.LEARNING_RATE)

    save_path = os.path.join(checkpoints_dir, f"{run_name}.pth")

    model = build_model(
        model_type=model_type,
        in_channels=config.IN_CHANNELS,
        hidden_channels=hidden_channels,
        out_channels=config.OUT_CHANNELS,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_metric = 0.0
    best_train_acc = 0.0
    best_test_acc = 0.0
    final_train_acc = 0.0
    final_test_acc = 0.0
    t0 = time.perf_counter()

    for epoch in range(config.NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        final_train_acc, final_test_acc = train_acc, test_acc

        current = test_acc if config.SAVE_BEST_BASED_ON == "test" else train_acc
        if current > best_metric:
            best_metric = current
            best_train_acc = train_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0:
            print(
                f"  [{run_name}] Epoch {epoch:03d}, Loss: {loss:.4f}, "
                f"Train: {train_acc:.4f}, Test: {test_acc:.4f}"
            )

    elapsed = time.perf_counter() - t0
    print(f"  [{run_name}] Done. Best test: {best_test_acc:.4f}, saved to {save_path}, time: {elapsed:.1f}s")

    return {
        "run_name": run_name,
        "model_type": model_type,
        "num_layers": num_layers,
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "learning_rate": lr,
        "best_train_acc": round(best_train_acc, 4),
        "best_test_acc": round(best_test_acc, 4),
        "final_train_acc": round(final_train_acc, 4),
        "final_test_acc": round(final_test_acc, 4),
        "time_sec": round(elapsed, 1),
        "save_path": save_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment presets from config.EXPERIMENT_PRESETS")
    parser.add_argument(
        "--presets",
        type=str,
        default=None,
        help="Comma-separated indices to run (e.g. 0,1,2). Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to append results CSV. Default: DATA_ROOT/experiment_results.csv",
    )
    args = parser.parse_args()

    presets = list(config.EXPERIMENT_PRESETS)
    if args.presets is not None:
        indices = [int(x.strip()) for x in args.presets.split(",")]
        presets = [presets[i] for i in indices if 0 <= i < len(presets)]
    if not presets:
        print("No presets to run.")
        return

    checkpoints_dir = os.path.join(config.DATA_ROOT, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(config.DATA_ROOT, "experiment_results.csv")

    # 固定划分，保证所有实验用同一 train/test 划分
    torch.manual_seed(args.seed)
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
    print(f"Device: {device}, presets: {len(presets)}, seed: {args.seed}")

    results = []
    for i, preset in enumerate(presets):
        print(f"\n--- Experiment {i + 1}/{len(presets)}: {preset['run_name']} ---")
        res = run_one_experiment(
            preset, train_loader, test_loader, device, checkpoints_dir
        )
        results.append(res)

    # 写 CSV（表头与第一次写入一致）
    fieldnames = [
        "run_name", "model_type", "num_layers", "hidden_channels", "dropout",
        "learning_rate", "best_train_acc", "best_test_acc",
        "final_train_acc", "final_test_acc", "time_sec", "save_path", "timestamp",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(results)
    print(f"\nResults appended to: {csv_path}")


if __name__ == "__main__":
    main()
