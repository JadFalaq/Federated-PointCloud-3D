from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from tqdm import tqdm

from feastnet_fl.federated.fedavg import weighted_fedavg
from feastnet_fl.training.trainer import evaluate, train_one_epoch


def run_federated_training(
    global_model,
    client_loaders: Dict[str, object],
    val_loader,
    device: torch.device,
    num_classes: int,
    rounds: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_path: str = "",
    show_progress: bool = True,
) -> Tuple[object, List[Dict[str, float]]]:
    history: List[Dict[str, float]] = []
    best_state = deepcopy(global_model.state_dict())
    best_accuracy = float("-inf")

    round_bar = tqdm(range(1, rounds + 1), desc="Federated Rounds") if show_progress else range(1, rounds + 1)
    for round_index in round_bar:
        client_updates = []
        client_losses = []

        client_items = list(client_loaders.items())
        client_bar = tqdm(client_items, desc=f"Round {round_index} Clients", leave=False) if show_progress else client_items
        for client_name, client_loader in client_bar:
            local_model = deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            for _ in range(local_epochs):
                loss = train_one_epoch(
                    local_model,
                    client_loader,
                    optimizer,
                    device,
                    show_progress=show_progress,
                )
                client_losses.append(loss)

            client_updates.append((deepcopy(local_model.state_dict()), len(client_loader.dataset)))

        aggregated_state = weighted_fedavg(client_updates)
        global_model.load_state_dict(aggregated_state)

        val_metrics = evaluate(
            global_model,
            val_loader,
            device,
            num_classes=num_classes,
            show_progress=show_progress,
        )
        record = {
            "round": float(round_index),
            "client_loss": float(sum(client_losses) / max(len(client_losses), 1)),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_class_accuracy": float(val_metrics["macro_class_accuracy"]),
        }
        history.append(record)

        log_msg = (
            f"Round {round_index}/{rounds} | "
            f"client_loss={record['client_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_acc={record['val_accuracy']:.4f}"
        )
        if show_progress:
            round_bar.write(log_msg)
        else:
            print(log_msg, flush=True)

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_state = deepcopy(global_model.state_dict())
            if checkpoint_path:
                path = Path(checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)

    global_model.load_state_dict(best_state)
    return global_model, history
