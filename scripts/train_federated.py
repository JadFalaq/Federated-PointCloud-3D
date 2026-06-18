from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feastnet_fl.config import load_config
from feastnet_fl.data.loaders import build_dataset, build_loader, build_loaders_from_splits
from feastnet_fl.data.prepare import prepare_experiment_splits
from feastnet_fl.evaluation.cross_client import evaluate_cross_client, load_client_splits
from feastnet_fl.evaluation.metrics import count_parameters
from feastnet_fl.evaluation.reporting import plot_history, save_json
from feastnet_fl.federated.simulator import run_federated_training
from feastnet_fl.training.trainer import evaluate
from feastnet_fl.utils.experiment import build_teacher_model, get_device, load_saved_splits
from feastnet_fl.utils.seed import set_seed

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated training entry point for FeaStNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    split_root = PROJECT_ROOT / config["paths"]["split_root"]
    if not (split_root / "train.csv").exists():
        prepare_experiment_splits(config)
    global_splits = load_saved_splits(config)

    scenario = config["federated"].get("scenario", "non_iid")
    scenario_root = split_root / scenario
    client_csv_paths = sorted(scenario_root.glob("*.csv"))
    if not client_csv_paths:
        prepare_experiment_splits(config)
        client_csv_paths = sorted(scenario_root.glob("*.csv"))

    labels = sorted(pd.concat(list(global_splits.values()), ignore_index=True)["label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    client_loaders = {}
    for client_csv_path in client_csv_paths:
        client_df = pd.read_csv(client_csv_path)
        dataset = build_dataset(
            metadata=client_df,
            metadata_csv=str(PROJECT_ROOT / config["paths"]["metadata_csv"]),
            pointcloud_root=str(PROJECT_ROOT / config["paths"]["pointcloud_root"]),
            label_to_index=label_to_index,
            num_points=config["sampling"]["num_points"],
            normalize=config["sampling"]["normalize"],
            augment_train=config["sampling"]["augment_train"],
            use_rgb=config["model"]["use_rgb"],
            stage="train",
        )
        client_loaders[client_csv_path.stem] = build_loader(
            dataset=dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
        )

    global_loaders = build_loaders_from_splits(
        splits={"val": global_splits["val"], "test": global_splits["test"]},
        metadata_csv=str(PROJECT_ROOT / config["paths"]["metadata_csv"]),
        pointcloud_root=str(PROJECT_ROOT / config["paths"]["pointcloud_root"]),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        num_points=config["sampling"]["num_points"],
        normalize=config["sampling"]["normalize"],
        augment_train=False,
        use_rgb=config["model"]["use_rgb"],
        label_to_index=label_to_index,
    )

    device = get_device(config["training"]["device"])
    model = build_teacher_model(config).to(device)

    checkpoint_path = PROJECT_ROOT / config["paths"]["output_root"] / "checkpoints" / f"feastnet_federated_{scenario}_best.pt"
    model, history = run_federated_training(
        global_model=model,
        client_loaders=client_loaders,
        val_loader=global_loaders["val"],
        device=device,
        num_classes=config["experiment"]["num_classes"],
        rounds=config["federated"]["rounds"],
        local_epochs=config["federated"]["local_epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        checkpoint_path=str(checkpoint_path),
    )
    test_metrics = evaluate(
        model=model,
        loader=global_loaders["test"],
        device=device,
        num_classes=config["experiment"]["num_classes"],
    )

    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    plot_history(history, str(output_root / "figures" / f"federated_{scenario}_convergence.png"))
    save_json(
        {
            "experiment": config["experiment"]["name"],
            "setup": f"federated_{scenario}",
            "test_metrics": test_metrics,
            "num_parameters": count_parameters(model),
            "history": history,
            "clients": list(client_loaders.keys()),
        },
        str(output_root / "reports" / f"federated_{scenario}_results.json"),
    )

    if scenario == "non_iid":
        client_splits = load_client_splits(str(split_root), scenario="non_iid")
        client_class_assignments = {
            client_name: set(client_df["label"].unique())
            for client_name, client_df in client_splits.items()
        }
        cross_client_report = evaluate_cross_client(
            model=model,
            test_metadata=global_splits["test"],
            label_to_index=label_to_index,
            client_class_assignments=client_class_assignments,
            metadata_csv=str(PROJECT_ROOT / config["paths"]["metadata_csv"]),
            pointcloud_root=str(PROJECT_ROOT / config["paths"]["pointcloud_root"]),
            device=device,
            num_classes=config["experiment"]["num_classes"],
            num_points=config["sampling"]["num_points"],
            normalize=config["sampling"]["normalize"],
            use_rgb=config["model"]["use_rgb"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
        )
        save_json(cross_client_report, str(output_root / "reports" / "cross_client_analysis.json"))
        print(
            "Cross-client mean accuracy on absent classes: "
            f"{cross_client_report['summary']['mean_accuracy_on_absent_classes']:.4f}"
        )

    print(f"Federated training completed for scenario: {scenario}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
