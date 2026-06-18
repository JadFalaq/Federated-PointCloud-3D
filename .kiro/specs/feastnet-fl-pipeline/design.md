# Design Document

## Overview

Pipeline complet de classification de nuages de points 3D avec FeaStNet (Graph CNN) implémentant quatre paradigmes d'entraînement : baseline centralisée (C1), fédération horizontale IID (C2), fédération horizontale non-IID (C3), distillation de connaissances (C4), plus la fédération verticale (VFL), des notebooks de visualisation et des scripts d'analyse comparative.

L'architecture existante (`src/feastnet_fl/`) est conservée intégralement. Les nouveaux composants s'y greffent sans modifier le code existant.

---

## Architecture

### Structure des nouveaux fichiers

```
PROJET_FL/
├── src/feastnet_fl/
│   ├── federated/
│   │   ├── fedavg.py          (existant)
│   │   ├── simulator.py       (existant)
│   │   └── vfl.py             ← NOUVEAU
│   └── evaluation/
│       ├── metrics.py         (existant)
│       ├── reporting.py       (existant)
│       └── cross_client.py    ← NOUVEAU
├── scripts/
│   ├── train_centralized.py   (existant)
│   ├── train_federated.py     (existant)
│   ├── train_distillation.py  (existant)
│   ├── train_vfl.py           ← NOUVEAU
│   ├── sweep_distillation.py  ← NOUVEAU
│   └── compare_results.py     ← NOUVEAU
└── notebooks/
    ├── 01_dataset_exploration.ipynb   ← NOUVEAU
    ├── 02_preprocessing.ipynb         ← NOUVEAU
    ├── 03_centralized_training.ipynb  ← NOUVEAU
    ├── 04_federated_hfl.ipynb         ← NOUVEAU
    ├── 05_federated_vfl.ipynb         ← NOUVEAU
    ├── 06_distillation.ipynb          ← NOUVEAU
    └── 07_comparison_analysis.ipynb   ← NOUVEAU
```

---

## Composants

### 1. VFL — `src/feastnet_fl/federated/vfl.py`

**Principe** : partition verticale des attributs de chaque point. Entité A possède xyz, Entité B possède rgb. Aucune donnée brute ne circule entre entités.

**Classes** :

```python
class VFLEntityEncoder(nn.Module):
    """Encodeur local pour une entité VFL (A ou B).
    Prend in_channels=3 (xyz ou rgb), produit embedding de dimension embed_dim."""
    
class VFLServer(nn.Module):
    """Serveur VFL : reçoit eA et eB, concatène, classe."""
    # Couche FC : Linear(2*embed_dim, num_classes)

class VFLTrainer:
    """Orchestre une epoch VFL complète :
    1. entity_a.encode(xyz) → eA  (avec gradient)
    2. entity_b.encode(rgb) → eB  (avec gradient)
    3. server.classify([eA||eB]) → logits
    4. loss.backward() → gradients vers eA et eB
    5. Mise à jour séparée : optimizer_a, optimizer_b, optimizer_server
    """
```

**Flux de données** :
```
Batch → pos (xyz) → Encodeur_A → eA ──┐
      → x[:,3:] (rgb) → Encodeur_B → eB ──┤→ [eA||eB] → FC → logits → loss
                                          Server
```

**Isolation** : xyz et rgb ne sont jamais transmis entre entités. Seuls `eA` et `eB` (vecteurs `float32` de dimension `d`) circulent vers le serveur.

---

### 2. Cross-client analysis — `src/feastnet_fl/evaluation/cross_client.py`

Évalue le modèle global fédéré (C3) sur les classes absentes de chaque client lors de l'entraînement.

```python
def evaluate_cross_client(
    model, test_loader, label_to_index, client_class_assignments, device, num_classes
) -> Dict[str, Dict]:
    """Pour chaque client, identifie les classes absentes et calcule
    la macro accuracy du modèle global sur ces classes uniquement."""
```

---

### 3. Sweep distillation — `scripts/sweep_distillation.py`

Grid search sur α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × τ ∈ {1.0, 2.0, 4.0, 8.0} = 20 combinaisons.
Export CSV : `alpha, temperature, val_accuracy, val_macro_class_accuracy, val_loss`.

---

### 4. Comparateur — `scripts/compare_results.py`

Lit les JSON de résultats de C1, C2, C3, C4 (et VFL optionnel), produit :
- Tableau Markdown + CSV des métriques finales de test
- Figure PNG : courbes de convergence superposées (val_loss + val_accuracy)

---

### 5. Notebooks

| Notebook | Contenu |
|---|---|
| `01_dataset_exploration` | Stats du CSV, distribution des classes, visualisation 3D de nuages PLY avec matplotlib |
| `02_preprocessing` | Sous-échantillonnage, normalisation, augmentation — avant/après côte à côte |
| `03_centralized_training` | Lancement C1, courbes d'entraînement, matrice de confusion |
| `04_federated_hfl` | C2 (IID) + C3 (non-IID), convergence par round, analyse cross-client |
| `05_federated_vfl` | Architecture VFL, flux d'embeddings, résultats VFL |
| `06_distillation` | C4, comparaison teacher vs student, sweep τ/α heatmap |
| `07_comparison_analysis` | Tableau C1–C4 + VFL, toutes courbes superposées, conclusion |

---

## Décisions de design

### VFLEntityEncoder — architecture

L'Entité A (xyz) utilise un sous-réseau FeaStConv : elle doit construire le kNN graph à partir de ses coordonnées spatiales. L'Entité B (rgb) ne possède pas de géométrie ; elle utilise un MLP sur les features de couleur avec pooling mean.

```
Entité A : FeaStConv(3 → 64) + BN + ReLU + FeaStConv(64 → embed_dim) + global_mean_pool
Entité B : Linear(3 → 64) + ReLU + Linear(64 → embed_dim) + (pooling via scatter_mean)
```

### Optimiseurs séparés en VFL

Trois optimiseurs Adam indépendants : `optimizer_a` (params Entité A), `optimizer_b` (params Entité B), `optimizer_server` (params FC serveur). La rétropropagation unique `loss.backward()` propage les gradients à travers la concaténation vers les deux encodeurs simultanément.

### Reproductibilité des notebooks

Chaque notebook appelle `set_seed(42)` en première cellule et utilise les configs YAML existantes via `load_config()`. Aucune valeur en dur.
