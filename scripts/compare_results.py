from pathlib import Path
import argparse
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from feastnet_fl.config import load_config
from feastnet_fl.evaluation.reporting import save_json


REPORT_SPECS = {
    "C1_centralized": ("centralized_results.json", "Centralized (C1)"),
    "C2_federated_iid": ("federated_iid_results.json", "Federated IID (C2)"),
    "C3_federated_non_iid": ("federated_non_iid_results.json", "Federated non-IID (C3)"),
    "C4_distillation": ("distillation_results.json", "Distillation (C4)"),
    "VFL": ("vfl_results.json", "Vertical FL (VFL)"),
}


def load_report(report_dir: Path, filename: str):
    path = report_dir / filename
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_row(key: str, label: str, report: dict) -> dict:
    test_metrics = report.get("test_metrics", {})
    summary = report.get("summary", {})
    history = report.get("history", [])
    return {
        "id": key,
        "configuration": label,
        "setup": report.get("setup", ""),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_macro_class_accuracy": test_metrics.get("macro_class_accuracy"),
        "test_loss": test_metrics.get("loss"),
        "num_parameters": report.get("num_parameters") or report.get("student_num_parameters"),
        "epochs_or_rounds": len(history),
        "best_val_accuracy": summary.get("best_val_accuracy"),
    }


def history_frame(label: str, report: dict) -> pd.DataFrame:
    history = report.get("history", [])
    if not history:
        return pd.DataFrame()
    frame = pd.DataFrame(history)
    x_key = "epoch" if "epoch" in frame.columns else "round"
    train_key = "train_loss" if "train_loss" in frame.columns else "client_loss"
    frame["configuration"] = label
    frame["x"] = frame[x_key]
    frame["train_metric"] = frame[train_key]
    return frame


def build_markdown_table(summary_df: pd.DataFrame) -> str:
    display_df = summary_df.copy()
    for column in ("test_accuracy", "test_macro_class_accuracy", "best_val_accuracy"):
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else "n/a"
            )
    if "num_parameters" in display_df.columns:
        display_df["num_parameters"] = display_df["num_parameters"].map(
            lambda value: f"{int(value):,}" if pd.notna(value) else "n/a"
        )
    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment reports and export summary tables.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = PROJECT_ROOT / config["paths"]["output_root"]
    report_dir = output_root / "reports"
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    missing = []
    for key, (filename, label) in REPORT_SPECS.items():
        report = load_report(report_dir, filename)
        if report is None:
            missing.append(filename)
            continue
        reports[label] = report

    if not reports:
        raise FileNotFoundError(f"No result reports found in {report_dir}")

    summary_rows = [
        extract_row(key, label, reports[label])
        for key, (filename, label) in REPORT_SPECS.items()
        if label in reports
    ]
    summary_df = pd.DataFrame(summary_rows)

    csv_path = report_dir / "comparison_summary.csv"
    md_path = report_dir / "comparison_summary.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown_table(summary_df), encoding="utf-8")

    history_frames = [history_frame(label, report) for label, report in reports.items()]
    history_frames = [frame for frame in history_frames if not frame.empty]
    if history_frames:
        history_df = pd.concat(history_frames, ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for label in history_df["configuration"].unique():
            subset = history_df[history_df["configuration"] == label]
            axes[0].plot(subset["x"], subset["train_metric"], marker="o", label=label)
            axes[0].plot(subset["x"], subset["val_loss"], linestyle="--", alpha=0.7)
            axes[1].plot(subset["x"], subset["val_accuracy"], marker="o", label=label)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch / Round")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=8)
        axes[1].set_title("Validation accuracy")
        axes[1].set_xlabel("Epoch / Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        overlay_path = figure_dir / "comparison_convergence_overlay.png"
        plt.savefig(overlay_path, dpi=150)
        plt.close()
    else:
        overlay_path = None

    save_json(
        {
            "experiment": config["experiment"]["name"],
            "reports_found": list(reports.keys()),
            "reports_missing": missing,
            "comparison_csv": str(csv_path),
            "comparison_markdown": str(md_path),
            "convergence_overlay_png": str(overlay_path) if overlay_path else None,
        },
        str(report_dir / "comparison_manifest.json"),
    )

    print("Comparison completed.")
    print(f"CSV : {csv_path}")
    print(f"Markdown : {md_path}")
    if overlay_path:
        print(f"Figure : {overlay_path}")
    if missing:
        print("Missing reports:")
        for filename in missing:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
