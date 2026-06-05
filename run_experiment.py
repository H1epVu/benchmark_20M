#!/usr/bin/env python3
"""
Run a single experiment: model + feature config + seed.

Usage:
  python run_experiment.py --model lightgcn --features none --seed 42
  python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42
  python run_experiment.py --model bpr_mf --features none --seed 42

  # SASRec — Direction 4 ablation
  python run_experiment.py --model sasrec --features none --injection none --seed 42
  python run_experiment.py --model sasrec --features llm_prof_mood --injection input --seed 42
  python run_experiment.py --model sasrec --features llm_prof_mood --injection ffn --seed 42
  python run_experiment.py --model sasrec --features llm_prof_mood --injection output --seed 42
  python run_experiment.py --model sasrec --features llm_prof_mood --injection all --seed 42
"""

import sys
import logging
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBED_DIM, LIGHTGCN_LAYERS, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, PATIENCE, FEATURE_CONFIGS, SEEDS,
    LIGHTGCL_SVD_Q, KAR_N_EXPERTS,
)
from data.dataset import InteractionData
from features.loader import FeatureLoader
from models.bpr_mf import BPRMF
from models.lightgcn import LightGCN, LightGCNSF
from models.xsimgcl import XSimGCL
from models.simgcl import SimGCL
from models.lightgcl import LightGCL as LightGCLModel
from models.kar import KAR
from models.sasrec import SASRec
from train import train_model, train_seq_model

SEQ_MODELS = {"sasrec"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    n_users: int,
    n_items: int,
    feature_dim: int = 0,
    norm_adj=None,
    injection_mode: str = "none",
):
    """Instantiate model by name."""
    if model_name == "bpr_mf":
        return BPRMF(n_users, n_items, EMBED_DIM)

    elif model_name == "lightgcn":
        model = LightGCN(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "lightgcn_sf":
        if feature_dim == 0:
            raise ValueError("lightgcn_sf requires features (feature_dim > 0)")
        model = LightGCNSF(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS, feature_dim)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "xsimgcl":
        model = XSimGCL(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "simgcl":
        model = SimGCL(n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS)
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "lightgcl":
        model = LightGCLModel(
            n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS,
            svd_q=LIGHTGCL_SVD_Q,
        )
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "kar":
        if feature_dim == 0:
            raise ValueError("kar requires features (feature_dim > 0)")
        model = KAR(
            n_users, n_items, EMBED_DIM, LIGHTGCN_LAYERS,
            feature_dim=feature_dim, n_experts=KAR_N_EXPERTS,
        )
        if norm_adj is not None:
            model.set_adj(norm_adj)
        return model

    elif model_name == "sasrec":
        return SASRec(
            n_users, n_items,
            embed_dim=EMBED_DIM,
            feature_dim=feature_dim,
            injection_mode=injection_mode,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["bpr_mf", "lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl", "kar", "sasrec"])
    parser.add_argument("--features",  type=str, default="none", choices=list(FEATURE_CONFIGS.keys()))
    parser.add_argument("--injection", type=str, default="none",
                        choices=["none", "input", "ffn", "output", "input+ffn", "input+output", "ffn+output", "all"],
                        help="Injection mode for SASRec (ignored for other models)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--lr",        type=float, default=LEARNING_RATE)
    parser.add_argument("--wd",        type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs",    type=int, default=NUM_EPOCHS)
    parser.add_argument("--seq-batch-size", type=int, default=512,
                        help="Batch size for sequential models (tune to fit VRAM)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision (AMP) — enabled by default on CUDA")
    parser.add_argument("--device",    type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if args.model in SEQ_MODELS:
        experiment_name = f"{args.model}__{args.injection}__{args.features}__seed{args.seed}"
    else:
        experiment_name = f"{args.model}__{args.features}__seed{args.seed}"
    logger.info(f"═══ Experiment: {experiment_name} ═══")
    logger.info(f"Device: {device}")

    set_seed(args.seed)

    # Load data
    logger.info("Loading interaction data...")
    data = InteractionData()

    # Load features
    feature_names = FEATURE_CONFIGS[args.features]
    feature_loader = FeatureLoader()
    feature_dim = feature_loader.get_feature_dim(feature_names) if feature_names else 0
    logger.info(f"Features: {args.features} → {feature_names} (dim={feature_dim})")

    # Build adjacency for GNN models
    norm_adj = None
    if args.model in ("lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl", "kar"):
        logger.info("Building normalized adjacency matrix...")
        norm_adj = data.get_norm_adj().to(device)

    # Build model
    model = build_model(
        args.model, data.n_users, data.n_items,
        feature_dim=feature_dim,
        norm_adj=norm_adj,
        injection_mode=args.injection,
    )

    # Set side features
    if args.model in ("lightgcn_sf", "kar") and feature_names:
        item_features = feature_loader.get_combined_tensor(feature_names, device=device)
        model.set_features(item_features)

    if args.model in SEQ_MODELS and feature_names:
        item_features = feature_loader.get_combined_tensor(feature_names, device=device)
        model.set_features(item_features)
        logger.info(f"Set item features: shape={tuple(item_features.shape)}")

    # Train
    if args.model in SEQ_MODELS:
        results = train_seq_model(
            model, data,
            device=device,
            lr=args.lr,
            weight_decay=args.wd,
            num_epochs=args.epochs,
            patience=PATIENCE,
            experiment_name=experiment_name,
            batch_size=args.seq_batch_size,
            use_amp=not args.no_amp,
        )
    else:
        results = train_model(
            model, data,
            device=device,
            lr=args.lr,
            weight_decay=args.wd,
            num_epochs=args.epochs,
            patience=PATIENCE,
            experiment_name=experiment_name,
        )

    logger.info(f"═══ Done: {experiment_name} ═══")
    return results


if __name__ == "__main__":
    main()
