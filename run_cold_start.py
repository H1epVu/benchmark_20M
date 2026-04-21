#!/usr/bin/env python3
"""
Cold-start analysis: compare multiple models across cold/medium/warm item buckets.

Usage:
  # Compare specific models (default: M0, M1, M3, M7)
  python run_cold_start.py --all-seeds

  # Compare custom model list
  python run_cold_start.py --models bpr_mf:none lightgcn:none lightgcn_sf:bert_title lightgcn_sf:llm_prof_mood --all-seeds

  # Single model, single seed
  python run_cold_start.py --models lightgcn_sf:llm_prof_mood --seed 123
"""

import sys
import json
import logging
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURE_CONFIGS, SEEDS, CHECKPOINT_DIR, RESULTS_DIR, TOP_K,
)
from data.dataset import InteractionData
from features.loader import FeatureLoader
from evaluate import evaluate_cold_start
from run_experiment import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default models to compare (matches README cold-start analysis intent)
DEFAULT_MODELS = [
    ("bpr_mf",      "none",          "M0  BPR-MF  "),
    ("lightgcn",    "none",          "M1  LightGCN"),
    ("lightgcn_sf", "bert_title",    "M3  BERT    "),
    ("lightgcn_sf", "llm_prof_mood", "M7  LLM p+m "),
]

BUCKETS = ["cold", "medium", "warm"]
DISPLAY_METRICS = ["NDCG@10", "NDCG@20", "NDCG@50", "Recall@10", "Recall@20", "Recall@50", "MRR"]


def load_and_evaluate(model_name, features_key, seed, data, feature_loader, device):
    """Load checkpoint and run cold-start evaluation for one seed."""
    experiment_name = f"{model_name}__{features_key}__seed{seed}"
    checkpoint_path = CHECKPOINT_DIR / experiment_name / "best_model.pt"

    if not checkpoint_path.exists():
        logger.warning(f"  Checkpoint not found: {checkpoint_path} — skipping")
        return None

    feature_names = FEATURE_CONFIGS[features_key]
    feature_dim = feature_loader.get_feature_dim(feature_names) if feature_names else 0

    norm_adj = None
    if model_name in ("lightgcn", "lightgcn_sf", "xsimgcl", "simgcl", "lightgcl", "kar"):
        norm_adj = data.get_norm_adj().to(device)

    model = build_model(model_name, data.n_users, data.n_items, feature_dim, norm_adj)

    if model_name in ("lightgcn_sf", "kar") and feature_names:
        item_features = feature_loader.get_combined_tensor(feature_names, device=device)
        model.set_features(item_features)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    return evaluate_cold_start(model, data, device=device, top_k=TOP_K)


def aggregate_seeds(seed_results):
    """Compute mean ± std across seeds for each bucket/metric."""
    buckets = list(seed_results[0].keys())
    metric_keys = [k for k in seed_results[0][buckets[0]].keys() if k != "n_eval"]

    agg = {}
    for bucket in buckets:
        agg[bucket] = {}
        for k in metric_keys:
            vals = [r[bucket][k] for r in seed_results if k in r.get(bucket, {})]
            if vals:
                agg[bucket][k] = float(np.mean(vals))
                agg[bucket][f"{k}_std"] = float(np.std(vals))
            else:
                agg[bucket][k] = 0.0
                agg[bucket][f"{k}_std"] = 0.0
        agg[bucket]["n_eval"] = seed_results[0][bucket].get("n_eval", 0)
        agg[bucket]["n_seeds"] = len(seed_results)
    return agg


def print_bucket_table(bucket, model_results, seeds_used):
    """Print one bucket's comparison table across models."""
    bucket_labels = {
        "cold":   "COLD    (112 items,   <10 train interactions)",
        "medium": "MEDIUM  (1,872 items, 10-50 train interactions)",
        "warm":   "WARM    (7,922 items, >50 train interactions)",
    }
    # Each metric cell: "0.0000000000±0.0000000000" = 25 chars, pad to 27
    CELL = 27
    MODEL_W = 16
    W = MODEL_W + 8 + len(DISPLAY_METRICS) * (CELL + 2)

    print(f"\n  Bucket: {bucket_labels[bucket]}")
    print("  " + "─" * W)

    # Header row
    header = f"  {'Model':<{MODEL_W}}  {'n_eval':>6}"
    for m in DISPLAY_METRICS:
        header += f"  {m:^{CELL}}"
    print(header)
    print("  " + "─" * W)

    for label, agg in model_results:
        if bucket not in agg:
            continue
        b = agg[bucket]
        n = b.get("n_eval", 0)
        row = f"  {label:<{MODEL_W}}  {n:>6}"
        for m in DISPLAY_METRICS:
            mean = b.get(m, 0.0)
            std  = b.get(f"{m}_std", 0.0)
            cell = f"{mean:.10f}±{std:.10f}"
            row += f"  {cell:^{CELL}}"
        print(row)
    print("  " + "─" * W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="MODEL:FEATURES",
        help="Models to compare, e.g. bpr_mf:none lightgcn_sf:bert_title"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true", help="Use all available seeds")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.seed is None and not args.all_seeds:
        parser.error("Provide --seed <int> or --all-seeds")

    # Parse model list
    if args.models:
        model_list = []
        for entry in args.models:
            parts = entry.split(":")
            if len(parts) != 2:
                parser.error(f"Invalid format '{entry}', expected MODEL:FEATURES")
            m, f = parts
            model_list.append((m, f, f"{m[:4]}:{f[:8]}"))
    else:
        model_list = DEFAULT_MODELS

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
    logger.info(f"Device: {device}")

    seeds = SEEDS if args.all_seeds else [args.seed]

    logger.info("Loading interaction data...")
    data = InteractionData()
    feature_loader = FeatureLoader()

    # Evaluate all models
    all_model_results = []  # [(label, agg_results)]
    for model_name, features_key, label in model_list:
        logger.info(f"--- {label.strip()} ({model_name} + {features_key}) ---")
        seed_results = []
        for seed in seeds:
            result = load_and_evaluate(model_name, features_key, seed, data, feature_loader, device)
            if result is not None:
                seed_results.append(result)

        if not seed_results:
            logger.warning(f"No checkpoints found for {label.strip()} — skipping")
            continue

        agg = aggregate_seeds(seed_results)
        all_model_results.append((label, agg))
        logger.info(f"  Collected {len(seed_results)} seed(s)")

    if not all_model_results:
        logger.error("No results collected — check checkpoints exist")
        return

    seeds_used = len(all_model_results[0][1].get("warm", {}).get("n_seeds", [1]) if False else
                     [r for _, r in all_model_results[:1]])

    # Print comparison tables
    W = 100
    print(f"\n{'═' * W}")
    print(f"  Cold-Start Analysis  |  {len(all_model_results)} model(s)  |  seeds: {seeds}")
    print(f"{'═' * W}")

    for bucket in BUCKETS:
        print_bucket_table(bucket, all_model_results, seeds)

    print(f"\n{'═' * W}\n")


    # Save results — each model into its own folder
    for (model_name, features_key, _), (label, agg) in zip(model_list, all_model_results):
        out_dir = RESULTS_DIR / f"{model_name}__{features_key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cold_start_results.json"
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
