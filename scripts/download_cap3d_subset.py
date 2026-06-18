from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
from remotezip import RemoteZip
from tqdm import tqdm

from feastnet_fl.config import load_config


CAP3D_SHAPENET_ZIP_URL = (
    "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/"
    "PointCloud_zips_ShapeNet/compressed_pcs_00.zip?download=true"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a balanced subset of CAP3D ShapeNet point clouds.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    metadata_csv = PROJECT_ROOT / config["paths"]["metadata_csv"]
    pointcloud_root = PROJECT_ROOT / config["paths"]["pointcloud_root"]
    pointcloud_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_csv)
    ids = metadata["id"].tolist()

    with RemoteZip(CAP3D_SHAPENET_ZIP_URL) as remote_zip:
        available_names = set(remote_zip.namelist())
        missing = []
        for object_id in tqdm(ids, desc="Downloading CAP3D subset"):
            remote_name = f"ShapeNet_pcs/{object_id}.ply"
            if remote_name not in available_names:
                missing.append(object_id)
                continue

            output_path = pointcloud_root / f"{object_id}.ply"
            if output_path.exists():
                continue

            with remote_zip.open(remote_name) as source, output_path.open("wb") as target:
                target.write(source.read())

    if missing:
        missing_path = PROJECT_ROOT / "data" / "processed" / "missing_cap3d_subset_ids.txt"
        missing_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"Missing files recorded in: {missing_path}")

    print(f"Subset download completed in: {pointcloud_root}")


if __name__ == "__main__":
    main()
