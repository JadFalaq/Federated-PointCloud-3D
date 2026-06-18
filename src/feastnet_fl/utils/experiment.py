from pathlib import Path

import pandas as pd
import torch

from feastnet_fl.models.feastnet import FeaStNetClassifier, StudentFeaStNet


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_teacher_model(config):
    model_cfg = config["model"]
    return FeaStNetClassifier(
        in_channels=model_cfg["in_channels"],
        hidden_channels=tuple(model_cfg["hidden_channels"]),
        heads=tuple(model_cfg["heads"]),
        k=model_cfg["k"],
        dropout=model_cfg["dropout"],
        num_classes=config["experiment"]["num_classes"],
    )


def build_student_model(config):
    model_cfg = config["model"]
    kd_cfg = config["knowledge_distillation"]
    return StudentFeaStNet(
        in_channels=model_cfg["in_channels"],
        hidden_channels=tuple(kd_cfg["student_hidden_channels"]),
        heads=tuple(kd_cfg["student_heads"]),
        k=max(model_cfg["k"] - 4, 4),
        dropout=max(model_cfg["dropout"] - 0.1, 0.1),
        num_classes=config["experiment"]["num_classes"],
    )


def load_saved_splits(config):
    split_root = Path(config["project_root"]) / config["paths"]["split_root"]
    return {
        "train": pd.read_csv(split_root / "train.csv"),
        "val": pd.read_csv(split_root / "val.csv"),
        "test": pd.read_csv(split_root / "test.csv"),
    }
