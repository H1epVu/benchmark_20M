"""
BPR training loop with validation, early stopping, checkpointing, and resume support.

Resume behavior:
  - If training_state.pt exists: resume from last saved epoch
  - If --epochs exceeds previous run: continue training from where it left off
  - Training state is ALWAYS preserved (never deleted) to allow extending training
"""

import time
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, PATIENCE,
    CHECKPOINT_DIR, RESULTS_DIR, TOP_K,
)
from data.dataset import InteractionData, get_train_loader
from evaluate import evaluate_model

logger = logging.getLogger(__name__)


def bpr_loss(pos_scores, neg_scores):
    """BPR loss: -log(sigmoid(pos - neg))."""
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def train_epoch(model, loader, optimizer, weight_decay, device):
    """Train one epoch with BPR loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for users, pos_items, neg_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        pos_scores, neg_scores, reg_loss = model(users, pos_items, neg_items)
        loss = bpr_loss(pos_scores, neg_scores) + weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def _save_training_state(checkpoint_dir, epoch, model, optimizer,
                         best_ndcg, best_metrics, best_epoch, patience_counter, history):
    """Save full training state for resume."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_ndcg": best_ndcg,
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "history": history,
    }, checkpoint_dir / "training_state.pt")


def _load_training_state(checkpoint_dir, model, optimizer, device):
    """Load training state for resume. Returns None if no checkpoint exists."""
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        return None

    state = torch.load(state_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])

    logger.info(f"  Resumed from epoch {state['epoch']} "
                f"(best NDCG@10: {state['best_ndcg']:.4f} at epoch {state['best_epoch']})")
    return state


def train_model(
    model,
    interaction_data: InteractionData,
    device: str = "cpu",
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    eval_every: int = 5,
    experiment_name: str = "default",
    resume: bool = True,
) -> Dict:
    """
    Full training loop with validation, early stopping, and resume support.

    Resume behavior:
      - If training_state.pt exists and epoch < num_epochs: continue training
      - If training_state.pt exists and epoch >= num_epochs: re-evaluate and save results
      - Training state is always preserved to allow extending with more epochs
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    loader = get_train_loader(interaction_data)

    best_ndcg = 0.0
    best_metrics = {}
    best_epoch = 0
    patience_counter = 0
    history = []
    start_epoch = 1

    checkpoint_dir = CHECKPOINT_DIR / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    if resume:
        state = _load_training_state(checkpoint_dir, model, optimizer, device)
        if state is not None:
            start_epoch = state["epoch"] + 1
            best_ndcg = state["best_ndcg"]
            best_metrics = state["best_metrics"]
            best_epoch = state["best_epoch"]
            patience_counter = state["patience_counter"]
            history = state["history"]

    if start_epoch > num_epochs:
        logger.info(f"All {num_epochs} epochs already completed for {experiment_name} — running test evaluation")
    else:
        logger.info(f"Training {experiment_name}: epochs {start_epoch}-{num_epochs}, lr={lr}, wd={weight_decay}")

    for epoch in range(start_epoch, num_epochs + 1):
        # Log epoch start
        epoch_start_time = datetime.now().strftime("%H:%M:%S")
        logger.info(f"[{epoch_start_time}] Epoch {epoch:3d}/{num_epochs} starting...")

        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, weight_decay, device)
        train_time = time.time() - t0

        log_entry = {"epoch": epoch, "train_loss": train_loss, "train_time": train_time}

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == 1:
            t0 = time.time()
            val_metrics = evaluate_model(model, interaction_data, split="val", device=device)
            eval_time = time.time() - t0

            ndcg10 = val_metrics.get("NDCG@10", 0)
            log_entry.update(val_metrics)
            log_entry["eval_time"] = eval_time

            epoch_end_time = datetime.now().strftime("%H:%M:%S")
            logger.info(
                f"[{epoch_end_time}] Epoch {epoch:3d}/{num_epochs} done | "
                f"Loss: {train_loss:.4f} | "
                f"NDCG@10: {ndcg10:.4f} | "
                f"Recall@20: {val_metrics.get('Recall@20', 0):.4f} | "
                f"Time: {train_time:.1f}s train + {eval_time:.1f}s eval"
            )

            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                best_metrics = val_metrics.copy()
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
                logger.info(f"  → New best NDCG@10: {ndcg10:.4f} (saved)")
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                    history.append(log_entry)
                    # Save state before breaking
                    _save_training_state(checkpoint_dir, epoch, model, optimizer,
                                         best_ndcg, best_metrics, best_epoch, patience_counter, history)
                    break
        else:
            epoch_end_time = datetime.now().strftime("%H:%M:%S")
            logger.info(
                f"[{epoch_end_time}] Epoch {epoch:3d}/{num_epochs} done | "
                f"Loss: {train_loss:.4f} | "
                f"Time: {train_time:.1f}s train (no eval)"
            )

        history.append(log_entry)

        # Save training state after every epoch for resume
        _save_training_state(checkpoint_dir, epoch, model, optimizer,
                             best_ndcg, best_metrics, best_epoch, patience_counter, history)

    # Load best model and evaluate on test set
    best_model_path = checkpoint_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    logger.info(f"Evaluating on test set (best model from epoch {best_epoch})...")
    test_metrics = evaluate_model(model, interaction_data, split="test", device=device)

    logger.info(f"═══ Test Results ({experiment_name}) ═══")
    for k, v in sorted(test_metrics.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    # Save final results
    results = {
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "total_epochs_trained": max(e["epoch"] for e in history) if history else 0,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "config": {
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "patience": patience,
        },
    }

    results_dir = RESULTS_DIR / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Training state is KEPT (not deleted) so training can be extended later
    # To continue training: increase --epochs and re-run
    logger.info(f"  Results saved. Training state preserved for potential continuation.")

    return results
