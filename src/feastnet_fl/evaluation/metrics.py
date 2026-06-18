from typing import Dict

import torch


def compute_classification_metrics(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    if targets.numel() == 0:
        return {"accuracy": 0.0, "macro_class_accuracy": 0.0, "num_examples": 0.0}

    accuracy = (targets == predictions).float().mean().item()
    class_accuracies = []
    for class_index in range(num_classes):
        mask = targets == class_index
        if mask.any():
            class_accuracies.append((predictions[mask] == targets[mask]).float().mean().item())

    macro_class_accuracy = sum(class_accuracies) / max(len(class_accuracies), 1)
    return {
        "accuracy": accuracy,
        "macro_class_accuracy": macro_class_accuracy,
        "num_examples": float(targets.numel()),
    }


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
