from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from feastnet_fl.distillation.losses import DistillationLoss
from feastnet_fl.training.trainer import evaluate


def train_distillation_epoch(
    student_model,
    teacher_model,
    loader,
    optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
    show_progress: bool = True,
) -> float:
    teacher_model.eval()
    student_model.train()
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Distill Batch", leave=False) if show_progress else loader
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_model(batch)
        student_logits = student_model(batch)
        loss = criterion(student_logits, teacher_logits, batch.y)
        loss.backward()
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if show_progress:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


def fit_distillation(
    teacher_model,
    student_model,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    num_classes: int,
    alpha: float,
    temperature: float,
    checkpoint_path: str = "",
    show_progress: bool = True,
) -> Tuple[object, List[Dict[str, float]]]:
    best_state = deepcopy(student_model.state_dict())
    best_accuracy = float("-inf")
    history: List[Dict[str, float]] = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Distillation Epochs") if show_progress else range(1, epochs + 1)
    for epoch in epoch_bar:
        train_loss = train_distillation_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            temperature=temperature,
            show_progress=show_progress,
        )
        val_metrics = evaluate(
            student_model,
            val_loader,
            device,
            num_classes=num_classes,
            show_progress=show_progress,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_class_accuracy": float(val_metrics["macro_class_accuracy"]),
            }
        )
        
        log_msg = (
            f"KD Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        if show_progress:
            epoch_bar.write(log_msg)
        else:
            print(log_msg, flush=True)

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_state = deepcopy(student_model.state_dict())
            if checkpoint_path:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)

    student_model.load_state_dict(best_state)
    return student_model, history
