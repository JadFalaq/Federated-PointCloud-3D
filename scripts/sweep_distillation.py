from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from feastnet_fl.config import load_config
from feastnet_fl.data.loaders import build_loaders_from_splits
from feastnet_fl.data.prepare import prepare_experiment_splits
from feastnet_fl.distillation.trainer import fit_distillation
from feastnet_fl.evaluation.reporting import save_json
from feastnet_fl.training.trainer import evaluate
from feastnet_fl.utils.experiment import build_student_model, build_teacher_model, get_device, load_saved_splits
from feastnet_fl.utils.seed import set_seed


ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
TEMPERATURE_VALUES = [1.0, 2.0, 4.0, 8.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search over distillation alpha and temperature.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs per sweep run (defaults to knowledge_distillation.sweep_epochs or training.epochs).",
    )
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

    kd_cfg = config.get("knowledge_distillation", {})
    sweep_epochs = args.epochs or kd_cfg.get("sweep_epochs", config["training"]["epochs"])
    device = get_device(config["training"]["device"])

    teacher_model = build_teacher_model(config).to(device)
    teacher_checkpoint = (
        PROJECT_ROOT / config["paths"]["output_root"] / "checkpoints" / "feastnet_centralized_best.pt"
    )
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_checkpoint}. Run train_centralized.py first."
        )
    teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device, weights_only=True))
    teacher_model.eval()

    rows = []
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    sweep_dir = output_root / "checkpoints" / "distillation_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    for alpha in ALPHA_VALUES:
        for temperature in TEMPERATURE_VALUES:
            print(f"Sweep alpha={alpha} temperature={temperature}", flush=True)
            student_model = build_student_model(config).to(device)
            optimizer = torch.optim.Adam(
                student_model.parameters(),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
            checkpoint_path = sweep_dir / f"student_a{alpha}_t{temperature}.pt"
            student_model, history = fit_distillation(
                teacher_model=teacher_model,
                student_model=student_model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                optimizer=optimizer,
                device=device,
                epochs=sweep_epochs,
                num_classes=config["experiment"]["num_classes"],
                alpha=alpha,
                temperature=temperature,
                checkpoint_path=str(checkpoint_path),
            )
            val_metrics = history[-1]
            test_metrics = evaluate(
                model=student_model,
                loader=loaders["test"],
                device=device,
                num_classes=config["experiment"]["num_classes"],
            )
            rows.append(
                {
                    "alpha": alpha,
                    "temperature": temperature,
                    "val_accuracy": val_metrics["val_accuracy"],
                    "val_macro_class_accuracy": val_metrics["val_macro_class_accuracy"],
                    "val_loss": val_metrics["val_loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_class_accuracy": test_metrics["macro_class_accuracy"],
                    "test_loss": test_metrics["loss"],
                }
            )

    sweep_df = pd.DataFrame(rows)
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "distillation_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)

    pivot = sweep_df.pivot(index="alpha", columns="temperature", values="val_accuracy")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2%", cmap="YlGnBu")
    plt.title("Distillation sweep — validation accuracy")
    plt.tight_layout()
    figure_path = output_root / "figures" / "distillation_sweep_heatmap.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150)
    plt.close()

    save_json(
        {
            "experiment": config["experiment"]["name"],
            "setup": "distillation_sweep",
            "alpha_values": ALPHA_VALUES,
            "temperature_values": TEMPERATURE_VALUES,
            "sweep_epochs": sweep_epochs,
            "results_csv": str(csv_path),
            "heatmap_png": str(figure_path),
            "best_row": sweep_df.sort_values("val_accuracy", ascending=False).iloc[0].to_dict(),
        },
        str(report_dir / "distillation_sweep_summary.json"),
    )

    print("Distillation sweep completed.")
    print(f"Results saved to: {csv_path}")
    print(f"Heatmap saved to: {figure_path}")


if __name__ == "__main__":
    main()
