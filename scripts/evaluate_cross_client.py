from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import torch

from feastnet_fl.config import load_config
from feastnet_fl.evaluation.cross_client import evaluate_cross_client, load_client_splits
from feastnet_fl.evaluation.reporting import save_json
from feastnet_fl.utils.experiment import build_teacher_model, get_device, load_saved_splits
from feastnet_fl.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a federated model on classes absent from each non-IID client."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path. Defaults to federated_non_iid best checkpoint.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    split_root = PROJECT_ROOT / config["paths"]["split_root"]
    global_splits = load_saved_splits(config)
    labels = sorted(pd.concat(list(global_splits.values()), ignore_index=True)["label"].unique())
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    client_splits = load_client_splits(str(split_root), scenario="non_iid")
    client_class_assignments = {
        client_name: set(client_df["label"].unique())
        for client_name, client_df in client_splits.items()
    }

    device = get_device(config["training"]["device"])
    model = build_teacher_model(config).to(device)
    checkpoint_path = args.checkpoint or str(
        PROJECT_ROOT
        / config["paths"]["output_root"]
        / "checkpoints"
        / "feastnet_federated_non_iid_best.pt"
    )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    report = evaluate_cross_client(
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

    output_path = PROJECT_ROOT / config["paths"]["output_root"] / "reports" / "cross_client_analysis.json"
    save_json(report, str(output_path))
    print("Cross-client analysis completed.")
    print(f"Report saved to: {output_path}")
    print(f"Mean accuracy on absent classes: {report['summary']['mean_accuracy_on_absent_classes']:.4f}")


if __name__ == "__main__":
    main()
