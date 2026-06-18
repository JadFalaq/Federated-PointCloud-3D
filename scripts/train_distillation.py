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
from feastnet_fl.distillation.trainer import fit_distillation
from feastnet_fl.evaluation.metrics import count_parameters
from feastnet_fl.evaluation.reporting import plot_history, save_json
from feastnet_fl.training.trainer import evaluate, fit
from feastnet_fl.utils.experiment import (
    build_student_model,
    build_teacher_model,
    get_device,
    load_saved_splits,
)
from feastnet_fl.utils.seed import set_seed

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge distillation entry point for FeaStNet.")
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
    teacher_model = build_teacher_model(config).to(device)
    teacher_checkpoint = (
        PROJECT_ROOT / config["paths"]["output_root"] / "checkpoints" / "feastnet_centralized_best.pt"
    )
    if teacher_checkpoint.exists():
        teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device, weights_only=True))
    else:
        teacher_optimizer = torch.optim.Adam(
            teacher_model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        teacher_model, _, _ = fit(
            model=teacher_model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            optimizer=teacher_optimizer,
            device=device,
            epochs=config["training"]["epochs"],
            num_classes=config["experiment"]["num_classes"],
            patience=config["training"]["early_stopping_patience"],
            checkpoint_path=str(teacher_checkpoint),
        )

    student_model = build_student_model(config).to(device)
    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    student_checkpoint = (
        PROJECT_ROOT / config["paths"]["output_root"] / "checkpoints" / "feastnet_student_distilled_best.pt"
    )
    student_model, history = fit_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
        num_classes=config["experiment"]["num_classes"],
        alpha=config["knowledge_distillation"]["alpha"],
        temperature=config["knowledge_distillation"]["temperature"],
        checkpoint_path=str(student_checkpoint),
    )
    test_metrics = evaluate(
        model=student_model,
        loader=loaders["test"],
        device=device,
        num_classes=config["experiment"]["num_classes"],
    )

    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    plot_history(history, str(output_root / "figures" / "distillation_convergence.png"))
    save_json(
        {
            "experiment": config["experiment"]["name"],
            "setup": "distillation",
            "teacher_checkpoint": str(teacher_checkpoint),
            "student_num_parameters": count_parameters(student_model),
            "test_metrics": test_metrics,
            "history": history,
            "alpha": config["knowledge_distillation"]["alpha"],
            "temperature": config["knowledge_distillation"]["temperature"],
        },
        str(output_root / "reports" / "distillation_results.json"),
    )
    print("Knowledge distillation completed.")
    print(f"Student test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
