# Implementation Plan

## Tasks

- [x] 1. Implémenter le module VFL (`src/feastnet_fl/federated/vfl.py`)
  - Créer `VFLEntityEncoder` : encodeur local pour entité A (xyz via FeaStConv) et entité B (rgb via MLP+pool)
  - Créer `VFLServer` : couche FC de fusion `[eA‖eB] → logits`
  - Créer `VFLTrainer` : orchestre l'epoch VFL avec 3 optimiseurs séparés, isolation stricte des données
  - Créer `run_vfl_training` : boucle d'entraînement complète avec validation, checkpointing, historique
  - **Files**: `src/feastnet_fl/federated/vfl.py`

- [-] 2. Créer le script d'entraînement VFL (`scripts/train_vfl.py`)
  - Charger la config YAML, préparer les splits, construire les DataLoaders
  - Instancier Entité_A, Entité_B, Serveur_VFL
  - Lancer `run_vfl_training`, évaluer sur le test set
  - Exporter JSON des métriques et courbes de convergence
  - **Files**: `scripts/train_vfl.py`

- [-] 3. Implémenter l'analyse cross-client (`src/feastnet_fl/evaluation/cross_client.py`)
  - Fonction `get_absent_classes_per_client` : identifie les classes absentes de chaque client non-IID
  - Fonction `evaluate_cross_client` : calcule macro accuracy par client sur ses classes absentes
  - Export JSON du rapport cross-client
  - **Files**: `src/feastnet_fl/evaluation/cross_client.py`

- [-] 4. Créer le script de sweep distillation (`scripts/sweep_distillation.py`)
  - Grid search α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} × τ ∈ {1.0, 2.0, 4.0, 8.0}
  - Pour chaque combinaison : entraîner StudentFeaStNet, enregistrer val_accuracy, val_macro_class_accuracy, val_loss
  - Export CSV récapitulatif et heatmap PNG (α vs τ, colorée par val_accuracy)
  - **Files**: `scripts/sweep_distillation.py`

- [-] 5. Créer le script de comparaison consolidée (`scripts/compare_results.py`)
  - Lire les JSON de résultats de C1, C2, C3, C4 (et VFL si disponible)
  - Produire tableau Markdown + CSV : configuration, accuracy globale, macro accuracy, nb paramètres, époques/rounds
  - Générer figure PNG avec courbes de convergence superposées (val_loss + val_accuracy)
  - Gérer les configurations manquantes avec avertissement
  - **Files**: `scripts/compare_results.py`

- [~] 6. Créer le notebook d'exploration du dataset (`notebooks/01_dataset_exploration.ipynb`)
  - Charger `labeled_dataset.csv`, afficher shape, colonnes, exemples
  - Visualisation de la distribution des classes (bar chart)
  - Chargement et visualisation 3D de 6 nuages de points .ply (scatter 3D matplotlib, coloré par rgb)
  - Statistiques : nb points par objet (min/max/mean), distribution des couleurs
  - **Files**: `notebooks/01_dataset_exploration.ipynb`

- [~] 7. Créer le notebook de prétraitement (`notebooks/02_preprocessing.ipynb`)
  - Démonstration visuelle du sous-échantillonnage (avant/après avec num_points=1024)
  - Démonstration de la normalisation (sphère unité)
  - Démonstration des augmentations : rotation, scale, jitter — comparaison côte à côte
  - Visualisation des splits train/val/test (distribution par classe)
  - Visualisation des partitions IID et non-IID des clients
  - **Files**: `notebooks/02_preprocessing.ipynb`

- [~] 8. Créer le notebook de baseline centralisée (`notebooks/03_centralized_training.ipynb`)
  - Instancier FeaStNetClassifier, afficher architecture et nombre de paramètres
  - Lancer l'entraînement C1 sur la config ultra_quick (rapide)
  - Afficher les courbes loss/accuracy epoch par epoch
  - Matrice de confusion sur le test set
  - Analyse des classes les mieux/moins bien classées
  - **Files**: `notebooks/03_centralized_training.ipynb`

- [~] 9. Créer le notebook fédération horizontale (`notebooks/04_federated_hfl.ipynb`)
  - Visualiser la répartition des classes par client (IID vs non-IID)
  - Lancer C2 (IID) et C3 (non-IID) sur config ultra_quick
  - Comparer les courbes de convergence C1 vs C2 vs C3
  - Analyse cross-client : accuracy sur classes absentes de chaque client
  - **Files**: `notebooks/04_federated_hfl.ipynb`

- [~] 10. Créer le notebook fédération verticale (`notebooks/05_federated_vfl.ipynb`)
  - Expliquer le protocole VFL avec schéma textuel ASCII
  - Visualiser la séparation xyz / rgb sur un exemple de nuage de points
  - Lancer l'entraînement VFL sur config ultra_quick
  - Comparer les embeddings eA et eB (PCA 2D)
  - Afficher les métriques finales
  - **Files**: `notebooks/05_federated_vfl.ipynb`

- [~] 11. Créer le notebook distillation (`notebooks/06_distillation.ipynb`)
  - Comparer architecture teacher vs student (nb paramètres, couches)
  - Lancer C4 sur config ultra_quick
  - Courbes de convergence teacher vs student
  - Visualiser l'effet de τ et α sur la loss (mini sweep 2×2)
  - **Files**: `notebooks/06_distillation.ipynb`

- [~] 12. Créer le notebook de comparaison finale (`notebooks/07_comparison_analysis.ipynb`)
  - Tableau récapitulatif C1/C2/C3/C4/VFL : accuracy, macro accuracy, paramètres
  - Courbes de convergence superposées pour toutes les configurations
  - Discussion des résultats et observations
  - **Files**: `notebooks/07_comparison_analysis.ipynb`

- [~] 13. Ajouter la config VFL au YAML et mettre à jour l'`__init__.py` des modules
  - Ajouter section `vfl` dans `configs/feastnet_ultra_quick_cpu.yaml` et `feastnet_base.yaml`
  - Exporter les nouveaux modules depuis les `__init__.py` de `federated/` et `evaluation/`
  - **Files**: `configs/feastnet_ultra_quick_cpu.yaml`, `configs/feastnet_base.yaml`, `src/feastnet_fl/federated/__init__.py`, `src/feastnet_fl/evaluation/__init__.py`
