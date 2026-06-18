"""
Vertical Federated Learning (VFL) module for point cloud classification.

Partition verticale des attributs :
  - Entité A : possède xyz (coordonnées spatiales)
  - Entité B : possède rgb (couleurs)

Seuls les embeddings eA et eB circulent vers le serveur — jamais les données brutes.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# Ensure src is on path when running standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from torch_geometric.nn import BatchNorm, FeaStConv, global_mean_pool

from feastnet_fl.evaluation.metrics import compute_classification_metrics, count_parameters
from feastnet_fl.models.feastnet import build_knn_edge_index


# ---------------------------------------------------------------------------
# VFLEntityEncoder
# ---------------------------------------------------------------------------

class VFLEntityEncoder(nn.Module):
    """Encodeur local pour une entité VFL.

    - Entité A (xyz, in_channels=3) : FeaStConv + kNN graph construit depuis pos
    - Entité B (rgb, in_channels=3) : MLP + global_mean_pool (pas de géométrie kNN)

    Les deux produisent un embedding de forme (batch_size, embed_dim).

    Parameters
    ----------
    in_channels : int
        Nombre de canaux d'entrée (3 pour xyz ou rgb).
    hidden_dim : int
        Dimension des couches cachées.
    embed_dim : int
        Dimension de l'embedding de sortie.
    k : int
        Nombre de plus proches voisins pour le kNN (Entité A uniquement).
    use_graph : bool
        True → architecture FeaStConv (Entité A / xyz).
        False → architecture MLP (Entité B / rgb).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        k: int = 8,
        use_graph: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.use_graph = use_graph

        if use_graph:
            # Entité A : deux couches FeaStConv avec BatchNorm
            self.conv1 = FeaStConv(in_channels, hidden_dim, heads=2)
            self.bn1 = BatchNorm(hidden_dim)
            self.conv2 = FeaStConv(hidden_dim, embed_dim, heads=4)
            self.bn2 = BatchNorm(embed_dim)
        else:
            # Entité B : MLP avec BatchNorm1d
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (N, in_channels)
            Features des points (xyz pour A, rgb pour B).
        pos : Tensor, shape (N, 3)
            Coordonnées 3D utilisées pour construire le kNN (Entité A).
            Pour Entité B ce paramètre est ignoré.
        batch : Tensor, shape (N,)
            Vecteur d'appartenance au graphe.

        Returns
        -------
        Tensor, shape (batch_size, embed_dim)
        """
        if self.use_graph:
            edge_index = build_knn_edge_index(pos, batch, k=self.k)
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            return global_mean_pool(x, batch)
        else:
            x = self.mlp(x)
            return global_mean_pool(x, batch)


# ---------------------------------------------------------------------------
# VFLServer
# ---------------------------------------------------------------------------

class VFLServer(nn.Module):
    """Serveur VFL : reçoit eA et eB, concatène, classifie.

    Architecture :
        Linear(2*embed_dim, 256) → ReLU → Dropout → Linear(256, num_classes)

    Parameters
    ----------
    embed_dim : int
        Dimension de chaque embedding (doit être identique pour eA et eB).
    num_classes : int
        Nombre de classes de sortie.
    dropout : float
        Taux de dropout entre les couches FC.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, eA: torch.Tensor, eB: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eA : Tensor, shape (batch_size, embed_dim)
        eB : Tensor, shape (batch_size, embed_dim)

        Returns
        -------
        Tensor, shape (batch_size, num_classes)

        Raises
        ------
        ValueError
            Si eA.shape[1] != eB.shape[1].
        """
        if eA.shape[1] != eB.shape[1]:
            raise ValueError(
                f"embed_dim mismatch: eA has {eA.shape[1]}, eB has {eB.shape[1]}"
            )
        combined = torch.cat([eA, eB], dim=1)
        return self.fc(combined)


# ---------------------------------------------------------------------------
# run_vfl_training
# ---------------------------------------------------------------------------

def run_vfl_training(
    entity_a: VFLEntityEncoder,
    entity_b: VFLEntityEncoder,
    server: VFLServer,
    train_loader,
    val_loader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_dir: str,
    patience: int = 10,
    show_progress: bool = True,
) -> Tuple[VFLEntityEncoder, VFLEntityEncoder, VFLServer, List[Dict], Dict]:
    """Boucle d'entraînement VFL complète.

    Trois optimiseurs Adam indépendants (optimizer_a, optimizer_b, optimizer_server).
    La rétropropagation unique `loss.backward()` propage les gradients vers les deux
    encodeurs simultanément via la concaténation.

    Isolation stricte :
      - entity_a ne voit que pos (xyz).
      - entity_b ne voit que batch.x[:, 3:] (rgb).
      - Le serveur ne touche jamais les données brutes.

    Parameters
    ----------
    entity_a : VFLEntityEncoder
        Encodeur de l'Entité A (xyz).
    entity_b : VFLEntityEncoder
        Encodeur de l'Entité B (rgb).
    server : VFLServer
        Serveur de fusion et classification.
    train_loader : DataLoader
        Loader d'entraînement.
    val_loader : DataLoader
        Loader de validation.
    device : torch.device
    num_classes : int
    epochs : int
    lr : float
    weight_decay : float
    checkpoint_dir : str
        Répertoire où sauvegarder le meilleur checkpoint (`vfl_best.pt`).
    patience : int
        Nombre d'époques sans amélioration avant l'arrêt anticipé.
    show_progress : bool
        Si True, affiche une barre de progression.

    Returns
    -------
    (entity_a, entity_b, server, history, summary)

    history : List[Dict]
        Clés : epoch, train_loss, val_loss, val_accuracy, val_macro_class_accuracy.
    summary : Dict
        Clés : best_epoch, best_val_accuracy, num_parameters.
    """
    entity_a = entity_a.to(device)
    entity_b = entity_b.to(device)
    server = server.to(device)

    optimizer_a = torch.optim.Adam(entity_a.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_b = torch.optim.Adam(entity_b.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_server = torch.optim.Adam(server.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_checkpoint_file = checkpoint_path / "vfl_best.pt"

    best_val_accuracy = float("-inf")
    best_epoch = 0
    no_improvement = 0
    best_state = {
        "entity_a": deepcopy(entity_a.state_dict()),
        "entity_b": deepcopy(entity_b.state_dict()),
        "server": deepcopy(server.state_dict()),
    }

    history: List[Dict] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="VFL Epochs") if show_progress else range(1, epochs + 1)
    for epoch in epoch_bar:
        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        entity_a.train()
        entity_b.train()
        server.train()

        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc="VFL Train Batch", leave=False) if show_progress else train_loader
        for batch in pbar:
            batch = batch.to(device)

            # Isolation des données : A voit uniquement xyz, B voit uniquement rgb
            pos = batch.pos  # (N, 3) — coordonnées spatiales

            if batch.x is not None and batch.x.shape[1] >= 6:
                rgb = batch.x[:, 3:]  # (N, 3) — canaux R, G, B
            else:
                rgb = torch.zeros(pos.shape[0], 3, device=device)

            y = batch.y

            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            optimizer_server.zero_grad()

            # Entité A : encode uniquement xyz (pos passé comme x ET pos)
            eA = entity_a(pos, pos, batch.batch)

            # Entité B : encode uniquement rgb (pos ignoré dans le forward MLP)
            eB = entity_b(rgb, pos, batch.batch)

            logits = server(eA, eB)
            loss = criterion(logits, y)
            loss.backward()

            optimizer_a.step()
            optimizer_b.step()
            optimizer_server.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if show_progress:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_samples, 1)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        val_metrics = _evaluate_vfl(
            entity_a,
            entity_b,
            server,
            val_loader,
            device,
            num_classes,
            criterion,
            show_progress=show_progress,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_class_accuracy": val_metrics["macro_class_accuracy"],
        }
        history.append(record)

        log_msg = (
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_acc={val_metrics['macro_class_accuracy']:.4f}"
        )
        if show_progress:
            epoch_bar.write(log_msg)
        else:
            print(log_msg, flush=True)

        # ------------------------------------------------------------------
        # Checkpoint & early stopping
        # ------------------------------------------------------------------
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            no_improvement = 0
            best_state = {
                "entity_a": deepcopy(entity_a.state_dict()),
                "entity_b": deepcopy(entity_b.state_dict()),
                "server": deepcopy(server.state_dict()),
            }
            torch.save(best_state, best_checkpoint_file)
        else:
            no_improvement += 1

        if no_improvement >= patience:
            log_stop = f"Early stopping at epoch {epoch} (patience={patience})"
            if show_progress:
                epoch_bar.write(log_stop)
            else:
                print(log_stop, flush=True)
            break

    # Restore best weights
    entity_a.load_state_dict(best_state["entity_a"])
    entity_b.load_state_dict(best_state["entity_b"])
    server.load_state_dict(best_state["server"])

    num_params = (
        count_parameters(entity_a)
        + count_parameters(entity_b)
        + count_parameters(server)
    )

    summary: Dict = {
        "best_epoch": float(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "num_parameters": float(num_params),
    }

    return entity_a, entity_b, server, history, summary


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_vfl(
    entity_a: VFLEntityEncoder,
    entity_b: VFLEntityEncoder,
    server: VFLServer,
    loader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
    show_progress: bool = True,
) -> Dict:
    """Évalue le pipeline VFL complet sur un DataLoader."""
    entity_a.eval()
    entity_b.eval()
    server.eval()

    all_targets = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="VFL Eval Batch", leave=False) if show_progress else loader
    for batch in pbar:
        batch = batch.to(device)

        pos = batch.pos
        if batch.x is not None and batch.x.shape[1] >= 6:
            rgb = batch.x[:, 3:]
        else:
            rgb = torch.zeros(pos.shape[0], 3, device=device)

        eA = entity_a(pos, pos, batch.batch)
        eB = entity_b(rgb, pos, batch.batch)
        logits = server(eA, eB)

        loss = criterion(logits, batch.y)
        preds = logits.argmax(dim=1)

        all_targets.append(batch.y.cpu())
        all_predictions.append(preds.cpu())

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if show_progress:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    predictions = torch.cat(all_predictions) if all_predictions else torch.empty(0, dtype=torch.long)

    metrics = compute_classification_metrics(targets, predictions, num_classes=num_classes)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics
