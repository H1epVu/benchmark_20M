#!/usr/bin/env python3
"""
Run full ablation table: all model × feature configs × seeds.

Usage:
  python run_ablation.py                    # run all experiments
  python run_ablation.py --quick            # quick test (1 seed, limited epochs)
  python run_ablation.py --model lightgcn_sf  # only LightGCN-SF ablations
"""

import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURE_CONFIGS, SEEDS, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ablation experiments: (model, features, label)
ABLATION_TABLE = [
    # Tier 1: Pure CF baselines
    ("bpr_mf",      "none",          "M0: BPR-MF (ID only)"),
    ("lightgcn",    "none",          "M1: LightGCN (ID only)"),

    ("simgcl",      "none",          "M1b: SimGCL (ID only)"),
    ("xsimgcl",     "none",          "M1c: XSimGCL (ID only)"),
    ("lightgcl",    "none",          "M1d: LightGCL (ID only)"),

    # Tier 2: Content-augmented LightGCN-SF
    ("lightgcn_sf", "genome",        "M2: + genome PCA"),
    ("lightgcn_sf", "bert_title",    "M3: + BERT title"),
    ("lightgcn_sf", "llm_profile",   "M4: + LLM profile"),
    ("lightgcn_sf", "llm_mood",      "M5: + LLM mood"),
    ("lightgcn_sf", "llm_themes",    "M6: + LLM themes"),
    ("lightgcn_sf", "llm_prof_mood", "M7: + LLM profile+mood"),
    ("lightgcn_sf", "llm_all",       "M8: + LLM all"),
    ("lightgcn_sf", "genome_llm",    "M9: + genome+mood+themes"),

    # Tier 3: LLM-for-RecSys methods
    ("kar",         "llm_prof_mood", "R3: KAR + LLM profile+mood"),
]


def run_single(model, features, seed, extra_args=None):
    """Run a single experiment as a subprocess."""
    cmd = [
        sys.executable, "run_experiment.py",
        "--model", model,
        "--features", features,
        "--seed", str(seed),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(Path(__file__).parent))


def collect_results():
    """Collect all results into a summary table."""
    summary = []
    for model, features, label in ABLATION_TABLE:
        seed_results = []
        for seed in SEEDS:
            exp_name = f"{model}__{features}__seed{seed}"
            result_file = RESULTS_DIR / exp_name / "results.json"
            if result_file.exists():
                with open(result_file) as f:
                    r = json.load(f)
                seed_results.append(r["test_metrics"])

        if seed_results:
            # Aggregate across seeds
            agg = {}
            metric_keys = [k for k in seed_results[0] if isinstance(seed_results[0][k], float)]
            for key in metric_keys:
                vals = [r[key] for r in seed_results]
                agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}

            summary.append({
                "label": label,
                "model": model,
                "features": features,
                "n_seeds": len(seed_results),
                "metrics": agg,
            })

    return summary


def print_summary(summary):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print(f"{'Experiment':<35} {'NDCG@10':>12} {'NDCG@20':>12} {'Recall@10':>12} {'Recall@20':>12} {'HR@10':>12} {'MRR':>12} {'Seeds':>6}")
    print("=" * 120)

    for entry in summary:
        m = entry["metrics"]
        def fmt(key):
            if key in m:
                return f"{m[key]['mean']:.4f}±{m[key]['std']:.4f}"
            return "N/A"

        print(f"{entry['label']:<35} {fmt('NDCG@10'):>12} {fmt('NDCG@20'):>12} "
              f"{fmt('Recall@10'):>12} {fmt('Recall@20'):>12} {fmt('HR@10'):>12} "
              f"{fmt('MRR'):>12} {entry['n_seeds']:>6}")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 seed, 20 epochs")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model")
    parser.add_argument("--collect-only", action="store_true", help="Only collect and print results")
    args = parser.parse_args()

    if args.collect_only:
        summary = collect_results()
        print_summary(summary)
        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "ablation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return

    seeds = [SEEDS[0]] if args.quick else SEEDS
    extra_args = ["--epochs", "20"] if args.quick else []

    experiments = ABLATION_TABLE
    if args.model:
        experiments = [(m, f, l) for m, f, l in experiments if m == args.model]

    total = len(experiments) * len(seeds)
    completed = 0

    for model, features, label in experiments:
        # Skip bert_title if not yet generated
        if features == "bert_title":
            bert_path = Path(__file__).parent.parent / "embedding_generator" / "output" / "bert_title_embeddings.npy"
            if not bert_path.exists():
                logger.warning(f"Skipping {label} — bert_title_embeddings.npy not found")
                completed += len(seeds)
                continue

        for seed in seeds:
            completed += 1
            exp_name = f"{model}__{features}__seed{seed}"
            result_file = RESULTS_DIR / exp_name / "results.json"
            if result_file.exists():
                logger.info(f"[{completed}/{total}] {label} (seed={seed}) — SKIPPED (complete)")
                continue
            logger.info(f"[{completed}/{total}] {label} (seed={seed})")
            result = run_single(model, features, seed, extra_args)
            if result.returncode != 0:
                logger.error(f"  FAILED: {label} seed={seed}")

    # Collect and print summary
    logger.info("Collecting results...")
    summary = collect_results()
    print_summary(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved to {RESULTS_DIR / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
