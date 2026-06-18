from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feastnet_fl.config import load_config
from feastnet_fl.data.prepare import prepare_experiment_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val/test and federated splits.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    split_paths = prepare_experiment_splits(config)
    print("Data splits prepared.")
    print(split_paths)


if __name__ == "__main__":
    main()
