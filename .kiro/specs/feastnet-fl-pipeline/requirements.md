# Requirements Document

## Introduction

Ce document définit les exigences pour le pipeline complet de classification de nuages de points 3D avec FeaStNet, couvrant quatre paradigmes d'entraînement : (C1) baseline centralisée, (C2) apprentissage fédéré horizontal IID, (C3) apprentissage fédéré horizontal non-IID, et (C4) distillation de connaissances. Il intègre également les deux composants manquants : l'apprentissage fédéré vertical (VFL) et les outils d'analyse comparative (sweep d'hyperparamètres, script de comparaison, analyse cross-client).

Le dataset est ShapeNet/CAP3D (50 074 objets, 50 classes, fichiers `.ply` avec xyz+rgb). Les implémentations existantes couvrent le pipeline de données, les modèles, l'entraînement centralisé, le HFL FedAvg, la distillation et l'évaluation. Les éléments à construire sont détaillés dans les exigences ci-dessous.

---

## Glossary

- **Pipeline** : l'ensemble du système de classification 3D, de la lecture des données à la production des métriques comparatives.
- **Dataset** : jeu de données ShapeNet/CAP3D composé de fichiers `.ply` portant des coordonnées spatiales (xyz) et des couleurs (rgb).
- **DataLoader** : composant de chargement des fichiers `.ply`, sous-échantillonnage, normalisation et augmentation.
- **Splitter** : composant de construction des partitions train/val/test et des partitions clients.
- **Sous-ensemble** : sous-sélection équilibrée et reproductible du Dataset complet.
- **FeaStNetClassifier** : modèle enseignant (Graph CNN), `in_channels=6`, `hidden_channels=(64,128,256)`, `heads=(4,4,8)`.
- **StudentFeaStNet** : modèle élève, au moins 4× moins de paramètres que FeaStNetClassifier (`hidden_channels=(32,64,64)`, `heads=(2,2,4)`).
- **Trainer** : composant d'entraînement centralisé avec `fit`, `evaluate`, `train_one_epoch`, early stopping et checkpointing.
- **FedAvg** : algorithme de moyenne pondérée des `state_dict` PyTorch, clé par clé.
- **Simulateur_HFL** : composant orchestrant les rounds de fédération horizontale.
- **Entité_A** : entité détenant uniquement les coordonnées xyz (VFL).
- **Entité_B** : entité détenant uniquement les valeurs rgb (VFL).
- **Serveur_VFL** : composant central VFL qui reçoit les embeddings, les concatène, effectue la classification et renvoie les gradients.
- **Encodeur_A** / **Encodeur_B** : troncs de réseau locaux de l'Entité_A et de l'Entité_B respectivement.
- **DistillationLoss** : fonction de perte combinant `CrossEntropy` et `KL_Divergence` pondérée par α et τ.
- **Comparateur** : script consolidant les résultats des configurations C1 à C4 et générant tableaux et courbes.
- **Sweep** : analyse systématique de l'impact des hyperparamètres α et τ sur les performances de distillation.
- **Analyseur_Cross_Client** : composant évaluant le modèle fédéré global sur des classes absentes de certains clients en non-IID.
- **Config** : fichier YAML de configuration d'une expérience (seed, chemins, hyperparamètres).
- **Round** : une itération de fédération comprenant l'entraînement local de tous les clients et une agrégation FedAvg.
- **Checkpoint** : sauvegarde du meilleur `state_dict` selon la précision de validation.
- **Précision_Globale** : proportion de prédictions correctes sur l'ensemble de test.
- **Précision_Macro_Classe** : moyenne non pondérée des précisions par classe sur l'ensemble de test.

---

## Requirements

### Requirement 1 : Pipeline de données

**User Story :** En tant que chercheur, je veux charger, sous-échantillonner, normaliser et augmenter les nuages de points ShapeNet, afin de produire des tenseurs homogènes prêts pour l'entraînement.

#### Critères d'acceptation

1. WHEN le DataLoader reçoit un fichier `.ply`, THE DataLoader SHALL extraire les coordonnées xyz et les valeurs rgb de chaque point.
2. WHEN le DataLoader sous-échantillonne un nuage de points, THE DataLoader SHALL retourner exactement `num_points` points, par sous-échantillonnage aléatoire sans remise si le nuage contient suffisamment de points, ou par duplication aléatoire sinon.
3. THE DataLoader SHALL normaliser les coordonnées xyz de sorte que le nuage centré soit inscrit dans une sphère de rayon 1,0 (norme maximale ≤ 1,0 + 1e-6).
4. WHILE le mode entraînement est actif, THE DataLoader SHALL appliquer les augmentations suivantes : rotation aléatoire autour de l'axe z, mise à l'échelle uniforme dans [0,9 ; 1,1], et bruit gaussien borné à ±0,05 sur xyz.
5. WHILE le mode évaluation ou test est actif, THE DataLoader SHALL retourner les nuages de points sans aucune augmentation.
6. THE Splitter SHALL construire les partitions train/val/test avec les ratios 0,70/0,15/0,15 par classe, de façon à ce que les ensembles train, val et test soient strictement disjoints.
7. THE Splitter SHALL garantir la reproductibilité des partitions via le paramètre `seed=42`.
8. THE Splitter SHALL sélectionner le Sous-ensemble de façon équilibrée : chaque classe représentée doit contenir le même nombre d'échantillons, à ±1 près.
9. IF un fichier `.ply` est illisible ou manquant, THEN THE DataLoader SHALL lever une exception `FileNotFoundError` avec le chemin du fichier concerné.

---

### Requirement 2 : Baseline centralisée (C1)

**User Story :** En tant que chercheur, je veux entraîner FeaStNetClassifier sur la totalité des données d'entraînement, afin d'obtenir une référence de performance maximale.

#### Critères d'acceptation

1. THE Trainer SHALL entraîner FeaStNetClassifier sur l'ensemble des données d'entraînement centralisées pendant au plus `epochs` époques.
2. WHEN la Précision_Globale sur l'ensemble de validation ne s'améliore pas pendant `early_stopping_patience` époques consécutives, THE Trainer SHALL arrêter l'entraînement.
3. WHEN la Précision_Globale sur l'ensemble de validation atteint un nouveau maximum, THE Trainer SHALL sauvegarder le Checkpoint correspondant.
4. THE Trainer SHALL rapporter, pour chaque époque, la perte d'entraînement, la perte de validation, la Précision_Globale et la Précision_Macro_Classe.
5. WHEN l'entraînement est terminé, THE Trainer SHALL charger le Checkpoint avec la meilleure Précision_Globale de validation et évaluer le modèle sur l'ensemble de test.
6. THE Trainer SHALL exporter les métriques finales (Précision_Globale, Précision_Macro_Classe, nombre de paramètres, courbe de convergence) au format JSON.

---

### Requirement 3 : Apprentissage fédéré horizontal IID (C2)

**User Story :** En tant que chercheur, je veux simuler un apprentissage fédéré avec données IID réparties entre 4 clients, afin d'évaluer FedAvg dans des conditions homogènes.

#### Critères d'acceptation

1. THE Splitter SHALL distribuer les données d'entraînement en mode IID par partitionnement round-robin sur 4 clients, de sorte que chaque client reçoive approximativement le même nombre d'exemples (±1 exemple près).
2. THE Simulateur_HFL SHALL exécuter `rounds` Rounds de fédération, chacun comprenant l'entraînement local de tous les clients pendant `local_epochs` époques.
3. WHEN le Simulateur_HFL démarre un Round, THE Simulateur_HFL SHALL distribuer une copie du modèle global à chaque client via `copy.deepcopy`.
4. WHEN un client termine son entraînement local, THE Simulateur_HFL SHALL collecter le `state_dict` mis à jour et la taille du jeu de données local de ce client.
5. THE FedAvg SHALL calculer la moyenne pondérée des paramètres flottants par la taille du jeu de données local de chaque client, clé par clé dans le `state_dict`.
6. THE FedAvg SHALL copier les buffers entiers (BatchNorm) depuis le client dont le jeu de données local est le plus grand.
7. IF une bibliothèque tierce de fédération est utilisée, THEN THE Pipeline SHALL signaler une erreur de configuration bloquante.
8. WHEN un Round est terminé, THE Simulateur_HFL SHALL évaluer le modèle global sur l'ensemble de validation et enregistrer les métriques du Round.
9. WHEN l'entraînement fédéré est terminé, THE Simulateur_HFL SHALL charger le meilleur Checkpoint et évaluer sur l'ensemble de test.

---

### Requirement 4 : Apprentissage fédéré horizontal non-IID (C3)

**User Story :** En tant que chercheur, je veux simuler un apprentissage fédéré avec partition sémantique non-IID entre 4 clients spécialisés, afin d'évaluer FedAvg dans des conditions hétérogènes réalistes.

#### Critères d'acceptation

1. THE Splitter SHALL assigner les classes aux clients selon la partition sémantique suivante : Client 1 = Mobilier (table, chair, bed, sofa, bench, bookshelf, file cabinet, cabinet), Client 2 = Transport (car, airplane, train, bus, motorbike, bicycle, watercraft, rocket), Client 3 = Technologie (laptop, telephone, camera, earphone, keyboard, display, printer, microphone, remote), Client 4 = Domestique (bottle, bowl, mug, can, knife, jar, basket, rifle, pistol, lamp, faucet, stove, washer, dishwasher, trash bin, mailbox, birdhouse, bag, cap, guitar, piano).
2. THE Splitter SHALL assigner les classes non listées aux clients existants en mode round-robin par ordre alphabétique de label.
3. THE Simulateur_HFL SHALL exécuter les Rounds de fédération non-IID en respectant les mêmes protocoles de communication (deepcopy, state_dict) que pour C2.
4. WHEN l'entraînement fédéré non-IID est terminé, THE Analyseur_Cross_Client SHALL évaluer le modèle global sur les classes absentes de chaque client et produire un rapport de Précision_Macro_Classe par client absent.
5. IF une classe du jeu de test n'est associée à aucun client lors de la partition sémantique, THEN THE Splitter SHALL assigner cette classe à un client existant via round-robin sans lever d'erreur.

---

### Requirement 5 : Apprentissage fédéré vertical (VFL)

**User Story :** En tant que chercheur, je veux implémenter la fédération verticale séparant xyz et rgb entre deux entités indépendantes, afin d'évaluer la classification sans partage de données brutes.

#### Critères d'acceptation

1. THE Entité_A SHALL recevoir uniquement le tenseur de coordonnées xyz (3 canaux) et SHALL NOT accéder aux valeurs rgb.
2. THE Entité_B SHALL recevoir uniquement le tenseur de couleurs rgb (3 canaux) et SHALL NOT accéder aux coordonnées xyz.
3. WHEN l'Entité_A traite un batch, THE Encodeur_A SHALL produire un vecteur d'embedding `eA ∈ R^d` sans transmettre les données xyz brutes au Serveur_VFL.
4. WHEN l'Entité_B traite un batch, THE Encodeur_B SHALL produire un vecteur d'embedding `eB ∈ R^d` sans transmettre les données rgb brutes au Serveur_VFL.
5. THE Serveur_VFL SHALL recevoir uniquement `eA` et `eB`, les concaténer en `[eA ∥ eB] ∈ R^(2d)`, et passer ce vecteur dans une couche fully-connected pour produire les logits de classification.
6. WHEN le Serveur_VFL calcule la perte, THE Serveur_VFL SHALL rétropropager les gradients vers l'Encodeur_A et l'Encodeur_B sans partager les paramètres internes de l'un avec l'autre.
7. THE Serveur_VFL SHALL mettre à jour uniquement ses propres paramètres (couche FC de fusion) lors de la rétropropagation.
8. WHEN l'entraînement VFL est terminé, THE Serveur_VFL SHALL exporter les métriques de test (Précision_Globale, Précision_Macro_Classe, nombre de paramètres total) au format JSON.
9. IF les vecteurs `eA` et `eB` n'ont pas la même dimension `d`, THEN THE Serveur_VFL SHALL lever une `ValueError` indiquant les dimensions reçues.

---

### Requirement 6: Distillation de connaissances (C4)

**User Story :** En tant que chercheur, je veux entraîner StudentFeaStNet guidé par FeaStNetClassifier (C1), afin d'obtenir un modèle léger avec des performances proches du modèle enseignant.

#### Critères d'acceptation

1. THE DistillationLoss SHALL calculer `L = (1-α)·CE(ŷ, y) + α·τ²·KL(softmax(zs/τ), softmax(zt/τ))` pour chaque batch.
2. THE DistillationLoss SHALL détacher les logits de l'enseignant (`teacher_logits.detach()`) avant tout calcul de gradient.
3. THE Trainer SHALL entraîner StudentFeaStNet avec DistillationLoss en utilisant FeaStNetClassifier (C1) chargé depuis son Checkpoint comme enseignant figé.
4. THE StudentFeaStNet SHALL posséder au moins 4× moins de paramètres entraînables que FeaStNetClassifier.
5. WHEN les hyperparamètres α ∈ {0,1 ; 0,3 ; 0,5 ; 0,7 ; 0,9} et τ ∈ {1,0 ; 2,0 ; 4,0 ; 8,0} sont variés, THE Sweep SHALL entraîner un modèle étudiant pour chaque combinaison et enregistrer la Précision_Globale, la Précision_Macro_Classe et la valeur de perte finale sur le jeu de validation.
6. WHEN le Sweep est terminé, THE Sweep SHALL exporter un tableau récapitulatif au format CSV avec les colonnes : `alpha`, `temperature`, `val_accuracy`, `val_macro_class_accuracy`, `val_loss`.
7. IF la valeur de DistillationLoss est `NaN` ou `Inf` pour un batch, THEN THE Trainer SHALL interrompre l'entraînement et enregistrer les valeurs de α et τ incriminées.

---

### Requirement 7 : Comparaison des configurations (C1–C4)

**User Story :** En tant que chercheur, je veux consolider les résultats de toutes les configurations dans un rapport unifié, afin de comparer objectivement les paradigmes d'entraînement.

#### Critères d'acceptation

1. THE Comparateur SHALL lire les fichiers de métriques JSON produits par les configurations C1, C2, C3, C4 et VFL depuis les chemins spécifiés dans la Config.
2. THE Comparateur SHALL produire un tableau comparatif avec les colonnes : `configuration`, `Précision_Globale`, `Précision_Macro_Classe`, `nombre_de_paramètres`, `époques_ou_rounds`.
3. THE Comparateur SHALL générer des courbes de convergence superposées (perte et précision de validation en fonction des époques/rounds) pour C1, C2, C3 et C4 dans un seul fichier image PNG.
4. WHEN des fichiers de métriques sont absents pour une configuration, THE Comparateur SHALL afficher un avertissement listant les configurations manquantes et produire le rapport partiel avec les données disponibles.
5. THE Comparateur SHALL sauvegarder le tableau comparatif au format CSV et au format Markdown dans le répertoire de sortie spécifié par la Config.

---

### Requirement 8: Analyse cross-client (non-IID)

**User Story :** En tant que chercheur, je veux évaluer le modèle fédéré global sur les classes que chaque client n'a jamais vues, afin de mesurer la capacité de généralisation du modèle global.

#### Critères d'acceptation

1. THE Analyseur_Cross_Client SHALL charger le modèle global issu de C3 depuis son Checkpoint.
2. THE Analyseur_Cross_Client SHALL identifier, pour chaque client non-IID, l'ensemble des classes absentes de son jeu de données d'entraînement.
3. WHEN l'ensemble des classes absentes d'un client est non vide, THE Analyseur_Cross_Client SHALL évaluer la Précision_Macro_Classe du modèle global uniquement sur ces classes et enregistrer le résultat.
4. THE Analyseur_Cross_Client SHALL exporter un rapport JSON avec, pour chaque client, la liste des classes absentes et la Précision_Macro_Classe correspondante.
5. IF l'ensemble des classes absentes d'un client est vide (toutes les classes présentes), THEN THE Analyseur_Cross_Client SHALL enregistrer `null` pour ce client dans le rapport sans lever d'erreur.

---

### Requirement 9 : Gestion de la configuration et de la reproductibilité

**User Story :** En tant que développeur, je veux que toutes les expériences soient entièrement configurables via des fichiers YAML et reproductibles, afin de faciliter la comparaison et la validation des résultats.

#### Critères d'acceptation

1. THE Pipeline SHALL charger tous les hyperparamètres (seed, ratios de split, num_points, learning_rate, rounds, α, τ, etc.) depuis un fichier YAML de Config sans valeurs codées en dur dans le code source.
2. THE Pipeline SHALL initialiser les générateurs aléatoires Python, NumPy et PyTorch avec le `seed` spécifié dans la Config avant toute opération aléatoire.
3. WHEN deux exécutions distinctes utilisent la même Config avec le même `seed`, THE Pipeline SHALL produire des partitions train/val/test identiques et des Checkpoints avec des valeurs de perte identiques à 1e-6 près.
4. THE Pipeline SHALL valider la Config au démarrage et SHALL lever une `ValueError` descriptive si un champ obligatoire est absent ou si une valeur est hors plage (ex. : ratios ne sommant pas à 1,0).
5. WHERE le paramètre `use_rgb` est désactivé dans la Config, THE DataLoader SHALL utiliser uniquement les coordonnées xyz (`in_channels=3`) et SHALL ignorer les valeurs rgb.

---

### Requirement 10 : Propriétés de correction formelles

**User Story :** En tant que développeur, je veux vérifier par tests à base de propriétés les invariants critiques du pipeline, afin de garantir l'absence de fuites de données et la correction des calculs.

#### Critères d'acceptation

1. FOR ALL listes de `state_dict` clients valides, THE FedAvg SHALL produire des poids agrégés tels que la somme des poids de pondération (`nk/n`) est exactement égale à 1,0 à 1e-7 près.
2. FOR ALL partitions générées par THE Splitter, les ensembles train, val et test SHALL être strictement disjoints : l'intersection de deux ensembles quelconques sur l'identifiant de fichier SHALL être vide.
3. FOR ALL valeurs de α ∈ [0,1] et τ > 0, WHEN les logits étudiant sont identiques aux logits enseignant, THE DistillationLoss SHALL retourner une valeur strictement inférieure à celle retournée lorsque les logits étudiant sont aléatoires.
4. FOR ALL batches traités par THE Serveur_VFL, le graphe de calcul de la perte SHALL contenir un chemin de gradient vers l'Encodeur_A et vers l'Encodeur_B, et SHALL NOT contenir de référence aux données xyz ou rgb brutes du batch.
5. FOR ALL Sous-ensembles générés par THE Splitter avec le même `seed`, le nombre d'échantillons par classe dans le Sous-ensemble SHALL être identique à ±1 près entre toutes les classes représentées.
6. FOR ALL nuages de points normalisés produits par THE DataLoader, la norme maximale des coordonnées xyz SHALL être ≤ 1,0 + 1e-6.
7. FOR ALL combinaisons (α, τ) testées par THE Sweep, la valeur de DistillationLoss pour une paire (logits_etudiant, logits_enseignant) identique SHA être comprise dans l'intervalle [0,0 ; +∞) et SHALL NOT être `NaN`.

