# PROJET_FL - FeaStNet Workspace

Ce depot prepare l'environnement de travail pour la partie `FeaStNet` du mini-projet sur la classification de nuages de points 3D, l'apprentissage federe manuel et la distillation de connaissances.

## Structure

```text
PROJET_FL/
|- configs/
|- data/
|  |- raw/
|  |- processed/
|  `- splits/
|- outputs/
|  |- checkpoints/
|  |- figures/
|  |- logs/
|  `- reports/
|- scripts/
`- src/feastnet_fl/
```

## Installation

Sous PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Points d'entree

```powershell
python scripts\train_centralized.py --config configs\feastnet_base.yaml
python scripts\train_federated.py --config configs\feastnet_base.yaml
python scripts\train_distillation.py --config configs\feastnet_base.yaml
```

## Donnees attendues

- `labeled_dataset.csv` sert de table de correspondance entre identifiants et labels.
- Les fichiers `.ply` doivent etre places dans `data/raw/pointclouds/` ou dans un chemin precise dans la configuration.
- Le pipeline final devra construire des splits reproductibles `train/val/test` et des partitions federes.
