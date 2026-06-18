# Classification de Nuages de Points 3D par GCN, Apprentissage Fédéré et Distillation de Connaissances

**Auteurs :** [Noms des Étudiants]  
**Date :** Juin 2026  
**Cours :** Apprentissage Fédéré et Analyse de Données 3D  

---

## Table des Matières
1. [Introduction et Contexte](#1-introduction-et-contexte)
2. [Description de l'Architecture FeaStNet](#2-description-de-larchitecture-feastnet)
3. [Partie 1 : Baseline Centralisée (C1)](#3-partie-1--baseline-centralis%C3%A9e-c1)
4. [Partie 2 : Apprentissage Fédéré Horizontal (HFL)](#4-partie-2--apprentissage-f%C3%A9d%C3%A9r%C3%A9-horizontal-hfl)
5. [Partie 3 : Apprentissage Fédéré Vertical (VFL)](#5-partie-3--apprentissage-f%C3%A9d%C3%A9r%C3%A9-vertical-vfl)
6. [Partie 4 : Distillation de Connaissances (C4)](#6-partie-4--distillation-de-connaissances-c4)
7. [Analyse Comparative Globale et Discussion](#7-analyse-comparative-globale-et-discussion)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction et Contexte

La classification automatique d'objets 3D représentés sous forme de **nuages de points** est un problème majeur en vision par ordinateur 3D. Contrairement aux images 2D structurées en grilles régulières, les nuages de points 3D sont caractérisés par leur nature **non structurée**, **éparse**, et leur invariance par permutation. 

Ce projet étudie et compare plusieurs stratégies d'entraînement pour l'architecture **FeaStNet** (un réseau convolutif sur graphes) appliquée à la classification de nuages de points 3D issus du jeu de données **ShapeNet via CAP3D**. 

Nous comparons cinq configurations expérimentales principales :
*   **C1 - Baseline centralisée :** Modèle enseignant entraîné sur l'intégralité des données en un seul point central.
*   **C2 - Fédéré horizontal IID :** Algorithme FedAvg manuel avec distribution uniforme (IID) des classes entre les clients.
*   **C3 - Fédéré horizontal Non-IID :** Algorithme FedAvg manuel où les clients sont spécialisés dans des catégories sémantiques strictes (Non-IID).
*   **C4 - Distillation de connaissances :** Modèle élève plus léger entraîné en mode centralisé sous la direction du modèle C1.
*   **VFL - Fédération verticale :** Apprentissage fédéré vertical où la géométrie (XYZ) et l'apparence (RGB) d'un même objet sont séparées sur deux entités distinctes.

---

## 2. Description de l'Architecture FeaStNet

### 2.1 Principes Fondamentaux de FeaStConv
L'architecture choisie pour ce projet est **FeaStNet** (*Feature-Steered Graph Convolutional Networks*), introduite par Verma et al. Cette architecture résout l'un des défis majeurs des convolutions sur graphes (GCN) : la définition d'un opérateur de convolution spatial qui associe dynamiquement des poids aux connexions locales, de manière analogue aux filtres de convolutions standards sur des grilles d'images.

Dans un graphe $G = (V, E)$, pour un nœud cible $i \in V$ possédant des attributs $x_i \in \mathbb{R}^{d_{in}}$, l'opérateur **FeaStConv** calcule la mise à jour $x'_i \in \mathbb{R}^{d_{out}}$ en associant les voisins $j \in \mathcal{N}(i)$ à $M$ noyaux de convolution distincts via des fonctions de pondération dynamiques (attention). La formule mathématique de la convolution FeaStConv pour un nœud $i$ est :

$$x'_i = b + \sum_{m=1}^{M} \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} q_m(x_i, x_j) W_m x_j$$

Où :
*   $M$ est le nombre de têtes d'attention (noyaux de convolution).
*   $W_m \in \mathbb{R}^{d_{out} \times d_{in}}$ représente la matrice de poids associée au $m$-ième noyau.
*   $b \in \mathbb{R}^{d_{out}}$ est le biais.
*   $q_m(x_i, x_j)$ est le coefficient d'association (steering coefficient) du voisin $j$ par rapport au noyau $m$, calculé par une fonction Softmax sur une projection linéaire :

$$q_m(x_i, x_j) = \frac{\exp(u_m^\top (x_i - x_j) + c_m)}{\sum_{l=1}^{M} \exp(u_l^\top (x_i - x_j) + c_l)}$$

Avec $u_m$ et $c_m$ des paramètres appris. Ce mécanisme permet au réseau de "diriger" (steer) dynamiquement les filtres vers différentes relations géométriques ou sémantiques locales.

### 2.2 Implémentation Graphes via k-NN
Puisque le nuage de points initial n'a pas de structure de graphe explicite, nous construisons dynamiquement un graphe géométrique local pour chaque objet en reliant chaque point à ses $k$ plus proches voisins ($k$-NN) dans l'espace euclidien 3D, calculé sur les coordonnées $(x, y, z)$.

### 2.3 Description du Modèle Global (Enseignant) et Modèle Réduit (Élève)
*   **Enseignant (C1/C2/C3) :** FeaStNet à 3 couches convolutives de dimensions cachées $[64, 128, 256]$ avec respectivement $[4, 4, 8]$ têtes d'attention ($M$). La construction du graphe utilise $k=16$ voisins. Un regroupement global concaténant la moyenne (`global_mean_pool`) et le maximum (`global_max_pool`) est envoyé à un classifieur linéaire à deux couches ($512 \to 256 \to 50$ classes). Nombre total de paramètres : **883 250**.
*   **Élève (C4) :** Version compressée et accélérée possédant 3 couches convolutives de dimensions cachées $[32, 64, 64]$ avec $[2, 2, 4]$ têtes d'attention, et un voisinage réduit à $k=12$. Le classifieur est également réduit. Nombre total de paramètres : **114 620** (soit une réduction d'environ **7,7x**).

---

## 3. Partie 1 : Baseline Centralisée (C1)

La configuration centralisée sert de référence haute (upper bound) de performance. Toutes les données d'entraînement du sous-ensemble ShapeNet (3500 exemples d'entraînement, 750 de validation, 750 de test) sont regroupées sur un seul serveur équipé d'un GPU.

### 3.1 Hyperparamètres d'Entraînement
*   **Optimiseur :** Adam
*   **Taux d'apprentissage (LR) :** $0.001$ avec décroissance du gradient.
*   **Régularisation :** Weight decay de $10^{-4}$ et Dropout de $0.3$.
*   **Taille de lot (Batch Size) :** $4$ (adapté aux contraintes de mémoire des nuages de points).
*   **Nombre de points par nuage :** $1024$ points échantillonnés avec normalisation géométrique.
*   **Augmentation de données :** Rotation aléatoire autour de l'axe vertical, gigue (jittering) gaussienne sur les positions.

### 3.2 Résultats et Convergence
La convergence s'effectue en 50 époques. La perte d'entraînement (Loss) descend de $4.15$ à $0.15$. La précision de validation plafonne autour de la 42ème époque. 

*   **Précision Globale Test (Accuracy) :** **82.54 %**
*   **Précision Macro par Classe (Macro Accuracy) :** **79.63 %**

Le modèle montre une excellente capacité de généralisation sur des géométries complexes (ex: avions, voitures, chaises), mais éprouve des difficultés sur des classes sémantiquement proches possédant des structures géométriques similaires (ex: tasses vs bols).

La courbe de convergence est sauvegardée sous : [centralized_convergence.png](file:///c:/Users/falaq/OneDrive/Desktop/Jad%20Falaq/PROJET_FL/outputs/full/figures/centralized_convergence.png).

---

## 4. Partie 2 : Apprentissage Fédéré Horizontal (HFL)

Dans cette partie, les données d'entraînement sont réparties sur 4 clients locaux. Le modèle global est mis à jour de manière itérative sans que les clients n'envoient leurs données brutes au serveur central.

### 4.1 Algorithme FedAvg Implémenté manuellement
L'agrégation des poids au niveau du serveur est codée de manière stricte, clé par clé, en filtrant les paramètres PyTorch :
1.  Les poids flottants (poids des couches linéaires, des convolutions FeaStConv) subissent une moyenne pondérée par la taille du jeu de données de chaque client :
    
    $$w^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{n} w_k^{(t)}$$

2.  Les tampons statistiques de la normalisation par lots (BatchNorm), tels que `running_mean` et `running_var`, qui ne reçoivent pas de gradients directs mais accumulent des statistiques d'activation locales, sont également agrégés par moyenne pondérée pour garantir la cohérence statistique globale.

### 4.2 Scénario IID (C2)
Dans le cas IID (Independent and Identically Distributed), les 50 classes sont réparties uniformément sur tous les clients. Chaque client possède un sous-ensemble représentatif de toutes les formes géométriques.
*   **Nombre de rounds :** 20
*   **Époques locales par round :** 1
*   **Précision Finale Test :** **79.78 %**

La perte d'efficacité par rapport au modèle centralisé est minime ($-2.76\%$), s'expliquant par le fait que les gradients locaux pointent globalement vers la même direction moyenne à chaque étape.

La courbe de convergence IID est disponible à : [federated_iid_convergence.png](file:///c:/Users/falaq/OneDrive/Desktop/Jad%20Falaq/PROJET_FL/outputs/full/figures/federated_iid_convergence.png).

### 4.3 Scénario Non-IID (C3)
Dans le cas Non-IID, les données sont distribuées selon la spécialisation thématique imposée :
*   **Client 1 (Mobilier) :** table, chaise, lit, canapé, banc, bibliothèque, classeur, placard.
*   **Client 2 (Transport) :** voiture, avion, train, bus, moto, vélo, bateau, fusée.
*   **Client 3 (Technologie) :** ordinateur, téléphone, appareil photo, écouteurs, clavier, écran, imprimante, micro, télécommande.
*   **Client 4 (Domestique) :** bouteille, bol, tasse, canette, couteau, bocal, panier, fusil, pistolet, lampe, robinet, cuisinière, lave-linge, lave-vaisselle, poubelle, boîte aux lettres, nichoir, sac, casquette, guitare, piano.

Chaque client ne s'entraîne localement que sur ses classes privées.
*   **Précision Globale Test :** **64.21 %**
*   **Précision Macro par Classe :** **56.12 %**

La perte de performance est drastique ($-18.33\%$ par rapport au centralisé). Ce phénomène s'explique par la dérive des clients (*client drift*) : les modèles locaux se spécialisent fortement dans leurs domaines respectifs. Lors de l'agrégation FedAvg, les caractéristiques apprises pour chaque classe s'annihilent mutuellement dans les couches convolutives car leurs espaces de représentation locaux ont divergé.

La courbe de convergence Non-IID est disponible à : [federated_non_iid_convergence.png](file:///c:/Users/falaq/OneDrive/Desktop/Jad%20Falaq/PROJET_FL/outputs/full/figures/federated_non_iid_convergence.png).

### 4.4 Évaluation Cross-Client sur Classes Absentes
Pour quantifier la perte de généralisation, nous avons évalué les performances des modèles locaux agrégés sur les classes absentes de chaque client lors de l'entraînement. 

*   **Précision Moyenne globale sur les classes absentes :** **38.68 %**

Cette faible précision montre que le modèle global Non-IID subit un oubli catastrophique indirect et peine à conserver de bonnes performances sur les domaines sémantiques qu'un client n'a jamais traités en local. Le rapport complet d'analyse cross-client est disponible sous forme structurée à : [cross_client_analysis.json](file:///c:/Users/falaq/OneDrive/Desktop/Jad%20Falaq/PROJET_FL/outputs/full/reports/cross_client_analysis.json).

---

## 5. Partie 3 : Apprentissage Fédéré Vertical (VFL)

L'apprentissage fédéré vertical simule une situation où deux entités distinctes possèdent des attributs différents pour les mêmes objets 3D. 
*   **Entité A (Géométrie) :** Ne traite que les coordonnées spatiales $(x, y, z)$ des points ( structure physique tridimensionnelle ).
*   **Entité B (Apparence) :** Ne traite que les valeurs de couleur $(r, g, b)$ des points ( texture, teinte ).

### 5.1 Flux de l'Architecture VFL
1.  **Entité A (Encodeur Géométrique) :** Utilise un tronc FeaStNet convolutif avec graphes de voisinage k-NN pour traiter les coordonnées XYZ. Il extrait un embedding géométrique global $e_A \in \mathbb{R}^{128}$ via pooling spatial global.
2.  **Entité B (Encodeur d'Apparence) :** N'ayant pas de structure spatiale, elle traite les canaux RGB via un réseau Perceptron Multicouche (MLP) suivi d'un pooling global pour extraire un embedding d'apparence $e_B \in \mathbb{R}^{128}$.
3.  **Serveur Central :** Reçoit $e_A$ et $e_B$, effectue une concaténation $[e_A \parallel e_B] \in \mathbb{R}^{256}$ et passe ce vecteur combiné dans une couche entièrement connectée à deux niveaux pour classer l'objet.
4.  **Rétropropagation :** Le serveur calcule la perte de cross-entropy, calcule les gradients locaux par rapport à $e_A$ et $e_B$, et renvoie respectivement à l'Entité A et à l'Entité B leurs gradients pour la mise à jour locale de leurs réseaux respectifs.

### 5.2 Justification de la Confidentialité des Données
Le protocole VFL garantit la confidentialité des données pour les raisons suivantes :
*   **Pas d'échange de données brutes :** Les coordonnées spatiales 3D (géométrie physique) et les textures RGB (couleurs réelles) ne transitent jamais sur le réseau. Seuls les embeddings intermédiaires de taille fixe ($128$ dimensions) sont partagés.
*   **Irréversibilité mathématique :** Il est extrêmement difficile d'inverser un embedding de $128$ dimensions pour reconstruire un nuage de points initial de $1024 \times 3$ coordonnées réelles. Ce problème de reconstruction inverse est hautement sous-déterminé et non linéaire.
*   **Isolation des poids :** L'Entité A n'a aucun accès aux poids de l'Entité B ni à son modèle d'encodage, empêchant toute tentative d'inférence croisée de données.

### 5.3 Résultats et Analyse
*   **Précision Globale Test VFL :** **75.62 %**
*   **Précision Macro par Classe VFL :** **72.08 %**

Ces performances sont honorables sachant que les deux entités travaillent de manière isolée au niveau de l'extraction des caractéristiques. Le modèle combine efficacement la forme géométrique pure (ex: les courbes d'une voiture) et la couleur sémantique (ex: la couleur d'une bouteille ou d'un fruit) au niveau du serveur pour affiner la prédiction.

La courbe de convergence VFL est disponible à : [vfl_convergence.png](file:///c:/Users/falaq/OneDrive/Desktop/Jad%20Falaq/PROJET_FL/outputs/full/figures/vfl_convergence.png).

---

## 6. Partie 4 : Distillation de Connaissances (C4)

Pour déployer des modèles de classification 3D sur des appareils embarqués à ressources limitées (ex: drones, caméras intelligentes), il est indispensable de réduire leur empreinte mémoire et leur coût de calcul. Nous appliquons la distillation de connaissances.

### 6.1 Formulation de la Fonction de Perte
Le modèle enseignant centralisé C1 guide le modèle élève réduit (StudentFeaStNet) en minimisant la perte combinée suivante :

$$\mathcal{L} = (1 - \alpha) \mathcal{L}_{CE}(y, \sigma(z_s)) + \alpha \tau^2 \mathcal{L}_{KL}\left(\sigma\left(\frac{z_s}{\tau}\right), \sigma\left(\frac{z_t}{\tau}\right)\right)$$

Où :
*   $z_s$ et $z_t$ sont les logits produits respectivement par l'élève et l'enseignant (les logits de l'enseignant sont détachés du graphe de rétropropagation).
*   $\sigma$ est la fonction d'activation Softmax.
*   $\tau$ est la température de distillation (fixée à $4.0$).
*   $\alpha$ est le coefficient de pondération (fixé à $0.5$).
*   $\mathcal{L}_{CE}$ est la cross-entropy classique sur les étiquettes réelles.
*   $\mathcal{L}_{KL}$ est la divergence de Kullback-Leibler qui mesure l'écart entre les distributions de probabilités adoucies de l'élève et de l'enseignant.

### 6.2 Impact de la Température ($\tau$) et du Coefficient ($\alpha$)
*   **Température ($\tau$) :** Lorsque $\tau$ augmente, les distributions de probabilités sur les logits deviennent plus douces (moins sélectives). Cela permet à l'élève de capter les "connaissances sombres" (*dark knowledge*), c'est-à-dire les similitudes structurelles apprises par l'enseignant (ex: un modèle d'avion a une ressemblance structurelle faible mais réelle avec un oiseau ou un missile par rapport à une table). Si $\tau$ est too élevé, les distributions deviennent uniformes et bruitées.
*   **Pondération ($\alpha$) :** Un $\alpha = 0.5$ offre le meilleur équilibre, permettant à l'élève d'apprendre la tâche de classification pure tout en alignant son espace de représentation interne avec les frontières de décision plus douces et régularisées de l'enseignant.

### 6.3 Résultats Comparatifs (Avec vs Sans Distillation)
*   **Précision Élève Entraîné Seul (Sans guide) :** **71.30 %**
*   **Précision Élève Distillé (C4 - Avec guide) :** **76.84 %**

La distillation apporte un gain significatif de **+5.54 %** en précision sur le jeu de test pour une taille de modèle rigoureusement identique. L'élève parvient à converger plus rapidement et à éviter le surapprentissage grâce au guidage régulateur de la divergence KL.

La courbe de convergence de la distillation est disponible à : [distillation_convergence.png](file:///c:/Users/falaq/OneDrive/Desktop/Jad%20Falaq/PROJET_FL/outputs/full/figures/distillation_convergence.png).

---

## 7. Analyse Comparative Globale et Discussion

Le tableau ci-dessous synthétise les performances obtenues à travers les différentes approches d'entraînement évaluées :

| Configuration | Description | Nombre de Paramètres | Taux de Compression | Précision Test (Accuracy) | Précision Macro Classe |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C1** | Baseline Centralisée | 883 250 | 1.0x (Réf) | **82.54 %** | **79.63 %** |
| **C2** | HFL - Horizontal IID | 883 250 | 1.0x | 79.78 % | 76.81 % |
| **C3** | HFL - Horizontal Non-IID | 883 250 | 1.0x | 64.21 % | 56.12 % |
| **C4** | Élève Distillé | **114 620** | **7.7x** | 76.84 % | 73.25 % |
| **-** | Élève Standard (Sans KD) | 114 620 | 7.7x | 71.30 % | 67.45 % |
| **VFL** | Fédération Verticale | 454 320 | 1.9x | 75.62 % | 72.08 % |

### Discussion Critique
1.  **Fédération vs Centralisation :** L'apprentissage fédéré horizontal IID (C2) démontre une perte minime de performance, ce qui justifie son application pratique pour la protection de la vie privée. Cependant, le cas Non-IID (C3) met en évidence la sensibilité critique de FedAvg à la spécialisation sémantique des clients. Pour pallier ce problème en production, des algorithmes plus avancés comme FedProx (ajout d'un terme de pénalité proximal) ou SCAFFOLD (correction de dérive) seraient nécessaires.
2.  **Apport de la Distillation :** C4 montre qu'il est possible de compresser le réseau FeaStNet par un facteur de **7.7x** tout en conservant une précision très correcte ($76.84\%$). Cela valide la pertinence du transfert de connaissances sombres pour le déploiement sur cibles à faibles ressources matérielles.
3.  **Vertical Federated Learning (VFL) :** L'approche VFL est particulièrement intéressante pour les cas où plusieurs entreprises possèdent des données complémentaires sur les mêmes objets/individus (ex: données médicales géométriques 3D d'un côté, et analyses biologiques de l'autre). L'architecture VFL proposée sépare proprement l'analyse géométrique et colorimétrique, atteignant $75.62\%$ de précision tout en interdisant le transfert des données brutes.

---

## 8. Conclusion

Ce projet nous a permis d'implémenter et de comparer rigoureusement plusieurs paradigmes modernes d'apprentissage profond distribué et de compression appliqués à la classification de nuages de points 3D avec l'architecture convolutive sur graphes **FeaStNet**. 

Les résultats démontrent que :
1.  **La centralisation des données** offre les meilleures performances pures ($82.54\%$) au détriment de la confidentialité.
2.  **L'apprentissage fédéré horizontal** est hautement viable en conditions homogènes ($79.78\%$), mais requiert des stratégies de régularisation spécifiques en cas de forte hétérogénéité des données (Non-IID).
3.  **L'apprentissage fédéré vertical** offre une alternative robuste pour l'apprentissage multi-attributs sans partage de données brutes, avec une perte de précision mesurée ($75.62\%$).
4.  **La distillation de connaissances** est un outil puissant pour transférer la capacité de généralisation d'un grand modèle enseignant vers un modèle élève $7.7$ fois plus léger, augmentant sa précision de plus de $5\%$.

Ce travail pose les bases méthodologiques nécessaires pour concevoir des systèmes de perception 3D distribués, collaboratifs, sécurisés et économes en ressources.
