"""Entry point script for Vertical Federated Learning (VFL) training.

Vertical partition:
  - Entity A: owns xyz coordinates
  - Entity B: owns rgb colours

Usage:
    python scripts/train_vfl.py --config configs/feastnet_ultra_quick_cpu.yaml
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from feastnet_fl.config import load_config
from feastnet_fl.data.loaders import build_loaders_from_splits
from feastnet_fl.data.prepare import prepare_experiment_splits
from feastnet_fl.evaluation.metrics import count_parameters
from feastnet_fl.evaluation.reporting import plot_history, save_json
from feastnet_fl.federated.vfl import (
    VFLEntityEncoder,
    VFLServer,
    _evaluate_vfl,
    run_vfl_training,
)
from feastnet_fl.utils.experiment import get_device, load_saved_splits
from feastnet_fl.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="VFL training entry point for FeaStNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config & reproducibility
    # ------------------------------------------------------------------
    config = load_config(args.config)
    set_seed(config["seed"])

    # ------------------------------------------------------------------
    # Data splits
    # ------------------------------------------------------------------
    split_root = PROJECT_ROOT / config["paths"]["split_root"]
    if not (split_root / "train.csv").exists():
        prepare_experiment_splits(config)
    splits = load_saved_splits(config)

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
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = get_device(config["training"]["device"])

    # ------------------------------------------------------------------
    # VFL hyper-parameters
    # ------------------------------------------------------------------
    vfl_cfg = config.get("vfl", {})
    embed_dim = vfl_cfg.get("embed_dim", 128)
    vfl_epochs = vfl_cfg.get("epochs", config["training"]["epochs"])

    # ------------------------------------------------------------------
    # VFL components
    # ------------------------------------------------------------------
    entity_a = VFLEntityEncoder(
        in_channels=3,
        hidden_dim=64,
        embed_dim=embed_dim,
        k=config["model"]["k"],
        use_graph=True,
    )
    entity_b = VFLEntityEncoder(
        in_channels=3,
        hidden_dim=64,
        embed_dim=embed_dim,
        k=config["model"]["k"],
        use_graph=False,
    )
    server = VFLServer(
        embed_dim=embed_dim,
        num_classes=config["experiment"]["num_classes"],
        dropout=config["model"]["dropout"],
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    checkpoint_dir = str(
        PROJECT_ROOT / config["paths"]["output_root"] / "checkpoints"
    )
    entity_a, entity_b, server, history, summary = run_vfl_training(
        entity_a=entity_a,
        entity_b=entity_b,
        server=server,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        num_classes=config["experiment"]["num_classes"],
        epochs=vfl_epochs,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_dir=checkpoint_dir,
        patience=config["training"]["early_stopping_patience"],
    )

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = _evaluate_vfl(
        entity_a=entity_a,
        entity_b=entity_b,
        server=server,
        loader=loaders["test"],
        device=device,
        num_classes=config["experiment"]["num_classes"],
        criterion=criterion,
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_root = PROJECT_ROOT / config["paths"]["output_root"]

    plot_history(history, str(output_root / "figures" / "vfl_convergence.png"))

    save_json(
        {
            "experiment": config["experiment"]["name"],
            "setup": "vfl",
            "summary": summary,
            "test_metrics": test_metrics,
            "history": history,
            "embed_dim": embed_dim,
            "entity_a_params": count_parameters(entity_a),
            "entity_b_params": count_parameters(entity_b),
            "server_params": count_parameters(server),
        },
        str(output_root / "reports" / "vfl_results.json"),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("VFL training completed.")
    print(f"Best validation accuracy : {summary['best_val_accuracy']:.4f}  (epoch {int(summary['best_epoch'])})")
    print(f"Test accuracy            : {test_metrics['accuracy']:.4f}")
    print(f"Test macro-class acc     : {test_metrics['macro_class_accuracy']:.4f}")
    print(f"embed_dim={embed_dim}  |  entity_a_params={count_parameters(entity_a):,}  |  entity_b_params={count_parameters(entity_b):,}  |  server_params={count_parameters(server):,}")


if __name__ == "__main__":
    main()
