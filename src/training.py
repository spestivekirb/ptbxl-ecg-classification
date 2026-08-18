"""Reusable PyTorch training utilities for multi-label ECG models."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingHistory:
    """Per-epoch losses recorded during training."""

    train_loss: list[float]
    val_loss: list[float]

    def as_dict(self) -> dict[str, list[float]]:
        return {"train_loss": self.train_loss, "val_loss": self.val_loss}


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed model-fitting run."""

    history: TrainingHistory
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Path | None


def select_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in descending order of preference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(signals)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = signals.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("data_loader did not yield any examples")
    return total_loss / total_examples


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""
    return _run_epoch(model, data_loader, criterion, device, optimizer)


def validate_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate one epoch without gradient updates."""
    return _run_epoch(model, data_loader, criterion, device)


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    scheduler=None,
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
    restore_best: bool = True,
    verbose: bool = True,
) -> TrainingResult:
    """Fit a model and retain the parameters with the lowest validation loss."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if resolved_checkpoint is not None:
        resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if resolved_checkpoint is not None:
                torch.save(best_state, resolved_checkpoint)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if verbose:
            print(
                f"Epoch {epoch:02d}/{num_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Validation loss: {val_loss:.4f}"
            )

        if (
            early_stopping_patience is not None
            and epoch - best_epoch >= early_stopping_patience
        ):
            if verbose:
                print(f"Early stopping after epoch {epoch}; best epoch: {best_epoch}")
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        history=TrainingHistory(train_losses, val_losses),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        checkpoint_path=resolved_checkpoint,
    )


def load_model_state(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load a weights-only state dictionary and move the model to ``device``."""
    state_dict = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    return model.to(device)
