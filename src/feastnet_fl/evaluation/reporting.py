import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def save_json(data: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def plot_history(history: List[Dict], output_path: str) -> None:
    if not history:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_key = "epoch" if "epoch" in history[0] else "round"
    train_key = "train_loss" if "train_loss" in history[0] else "client_loss"
    y_acc_key = "val_accuracy"

    epochs = [item[x_key] for item in history]
    train_loss = [item[train_key] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_accuracy = [item[y_acc_key] for item in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel(x_key.capitalize())
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.xlabel(x_key.capitalize())
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
