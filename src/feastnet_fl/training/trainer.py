from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from feastnet_fl.evaluation.metrics import count_parameters
from feastnet_fl.evaluation.metrics import compute_classification_metrics


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    criterion=None,
    show_progress: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    criterion = criterion or nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Train Batch", leave=False) if show_progress else loader
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if show_progress:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    criterion=None,
    show_progress: bool = True,
) -> Dict[str, float]:
    model.eval()
    all_targets = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0
    criterion = criterion or nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="Eval Batch", leave=False) if show_progress else loader
    for batch in pbar:
        batch = batch.to(device)
        logits = model(batch)
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


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    num_classes: int,
    patience: int = 10,
    checkpoint_path: str = "",
    show_progress: bool = True,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, float]]:
    history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    no_improvement = 0

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs") if show_progress else range(1, epochs + 1)
    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            show_progress=show_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=num_classes,
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

        if val_metrics["accuracy"] > best_metric:
            best_metric = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
            if checkpoint_path:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)
        else:
            no_improvement += 1

        if no_improvement >= patience:
            break

    model.load_state_dict(best_state)
    summary = {
        "best_epoch": float(best_epoch),
        "best_val_accuracy": float(best_metric),
        "num_parameters": float(count_parameters(model)),
    }
    return model, history, summary
