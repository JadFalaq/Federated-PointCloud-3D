from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
from torch import nn

from feastnet_fl.evaluation.metrics import compute_classification_metrics


def get_client_class_assignments(client_splits: Dict[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    return {
        client_name: set(client_df["label"].unique())
        for client_name, client_df in client_splits.items()
    }


def get_absent_classes_per_client(
    all_labels: List[str],
    client_class_assignments: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    absent = {}
    all_label_set = set(all_labels)
    for client_name, client_labels in client_class_assignments.items():
        absent[client_name] = sorted(all_label_set - client_labels)
    return absent


def load_client_splits(split_root: str, scenario: str = "non_iid") -> Dict[str, pd.DataFrame]:
    root = Path(split_root) / scenario
    if not root.exists():
        raise FileNotFoundError(f"Client split directory not found: {root}")
    return {path.stem: pd.read_csv(path) for path in sorted(root.glob("*.csv"))}


@torch.no_grad()
def evaluate_cross_client(
    model: nn.Module,
    test_metadata: pd.DataFrame,
    label_to_index: Dict[str, int],
    client_class_assignments: Dict[str, Set[str]],
    metadata_csv: str,
    pointcloud_root: str,
    device: torch.device,
    num_classes: int,
    num_points: int,
    normalize: bool,
    use_rgb: bool,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Dict]:
    from feastnet_fl.data.loaders import build_dataset, build_loader

    index_to_label = {index: label for label, index in label_to_index.items()}
    all_labels = sorted(label_to_index.keys())
    absent_classes = get_absent_classes_per_client(all_labels, client_class_assignments)
    per_client = []

    model.eval()
    for client_name, missing_labels in absent_classes.items():
        if not missing_labels:
            per_client.append(
                {
                    "client": client_name,
                    "present_classes": len(client_class_assignments[client_name]),
                    "absent_classes": 0,
                    "absent_class_names": [],
                    "accuracy_on_absent_classes": None,
                    "macro_class_accuracy": None,
                    "num_examples": 0,
                }
            )
            continue

        absent_indices = {label_to_index[label] for label in missing_labels}
        subset = test_metadata[test_metadata["label"].isin(missing_labels)].reset_index(drop=True)
        if subset.empty:
            per_client.append(
                {
                    "client": client_name,
                    "present_classes": len(client_class_assignments[client_name]),
                    "absent_classes": len(missing_labels),
                    "absent_class_names": missing_labels,
                    "accuracy_on_absent_classes": None,
                    "macro_class_accuracy": None,
                    "num_examples": 0,
                }
            )
            continue

        dataset = build_dataset(
            metadata=subset,
            metadata_csv=metadata_csv,
            pointcloud_root=pointcloud_root,
            label_to_index=label_to_index,
            num_points=num_points,
            normalize=normalize,
            augment_train=False,
            use_rgb=use_rgb,
            stage="test",
        )
        loader = build_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        all_targets = []
        all_predictions = []
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch).argmax(dim=1)
            all_targets.append(batch.y.cpu())
            all_predictions.append(predictions.cpu())

        targets = torch.cat(all_targets)
        predictions = torch.cat(all_predictions)
        metrics = compute_classification_metrics(targets, predictions, num_classes=num_classes)

        class_accuracies = []
        for class_index in absent_indices:
            mask = targets == class_index
            if mask.any():
                class_accuracies.append((predictions[mask] == targets[mask]).float().mean().item())

        per_client.append(
            {
                "client": client_name,
                "present_classes": len(client_class_assignments[client_name]),
                "absent_classes": len(missing_labels),
                "absent_class_names": missing_labels,
                "accuracy_on_absent_classes": metrics["accuracy"],
                "macro_class_accuracy": sum(class_accuracies) / max(len(class_accuracies), 1),
                "num_examples": int(metrics["num_examples"]),
            }
        )

    valid_scores = [
        row["accuracy_on_absent_classes"]
        for row in per_client
        if row["accuracy_on_absent_classes"] is not None
    ]
    return {
        "per_client": per_client,
        "summary": {
            "mean_accuracy_on_absent_classes": sum(valid_scores) / max(len(valid_scores), 1),
            "clients_evaluated": len(valid_scores),
        },
        "index_to_label": index_to_label,
    }
