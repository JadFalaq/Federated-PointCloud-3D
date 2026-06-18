from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feastnet_fl.config import load_config
from feastnet_fl.data.loaders import build_loaders_from_splits
from feastnet_fl.data.prepare import prepare_experiment_splits
from feastnet_fl.evaluation.metrics import count_parameters
from feastnet_fl.evaluation.reporting import plot_history, save_json
from feastnet_fl.training.trainer import evaluate, fit
from feastnet_fl.utils.experiment import build_teacher_model, get_device, load_saved_splits
from feastnet_fl.utils.seed import set_seed

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized training entry point for FeaStNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    split_root = PROJECT_ROOT / config["paths"]["split_root"]
    if not (split_root / "train.csv").exists():
        prepare_experiment_splits(config)
    splits = load_saved_splits(config)
    labels = sorted(pd.concat(list(splits.values()), ignore_index=True)["label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    loaders = build_loaders_from_splits(
        splits=splits,
        metadata_csv=str(PROJECT_ROOT / config["paths"]["metadata_csv"]),
        pointcloud_root=str(PROJECT_ROOT / config["paths"]["pointcloud_root"]),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        num_points=config["sampling"]["num_points"],
        normalize=config["sampling"]["normalize"],
        augment_train=config["sampling"]["augment_train"],
        use_rgb=config["model"]["use_rgb"],
        label_to_index=label_to_index,
    )

    device = get_device(config["training"]["device"])
    model = build_teacher_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_path = (
        PROJECT_ROOT / config["paths"]["output_root"] / "checkpoints" / "feastnet_centralized_best.pt"
    )
    model, history, summary = fit(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
        num_classes=config["experiment"]["num_classes"],
        patience=config["training"]["early_stopping_patience"],
        checkpoint_path=str(checkpoint_path),
    )
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        device=device,
        num_classes=config["experiment"]["num_classes"],
    )

    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    plot_history(history, str(output_root / "figures" / "centralized_convergence.png"))
    save_json(
        {
            "experiment": config["experiment"]["name"],
            "setup": "centralized",
            "summary": summary,
            "test_metrics": test_metrics,
            "num_parameters": count_parameters(model),
            "history": history,
        },
        str(output_root / "reports" / "centralized_results.json"),
    )
    print("Centralized training completed.")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
