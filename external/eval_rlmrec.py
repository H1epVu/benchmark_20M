#!/usr/bin/env python3
"""
Benchmark-compatible evaluation for saved RLMRec checkpoints.

Uses benchmark's evaluate.py (full-rank, all items) so metrics are
directly comparable to M models.

Must be run from external/RLMRec/ directory:
  cd /path/to/benchmark/external/RLMRec
  python ../eval_rlmrec.py --model lightgcn_gene --seed 42 --device cuda
"""

import sys
import json
import argparse
import torch
from pathlib import Path

# ── Absolute paths ───────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent   # external/
RLMREC_ENCODER = SCRIPT_DIR / "RLMRec" / "encoder"
BENCHMARK_ROOT = SCRIPT_DIR.parent

# ── Step 1: import benchmark modules first (before RLMRec pollutes sys.path) ─
# Remove '' so cwd (external/RLMRec/) doesn't shadow benchmark's data package
sys.path = [p for p in sys.path if p != ""]
sys.path.insert(0, str(BENCHMARK_ROOT))

from data.dataset import InteractionData   # benchmark's data/dataset.py
from evaluate import evaluate_model        # benchmark's evaluate.py

# Unregister 'config' so RLMRec can load its own config package next
for key in list(sys.modules.keys()):
    if key == "config" or key.startswith("config."):
        del sys.modules[key]
sys.path.remove(str(BENCHMARK_ROOT))

# ── Step 2: parse args early so RLMRec configurator reads correct values ────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model",   type=str, default="lightgcn_gene")
_parser.add_argument("--dataset", type=str, default="ml20m")
_parser.add_argument("--seed",    type=int, default=42)
_parser.add_argument("--device",  type=str, default="cuda")
_parser.add_argument("--split",   type=str, default="test")
_pre_args, _ = _parser.parse_known_args()

sys.argv = [sys.argv[0],
            "--model",   _pre_args.model,
            "--dataset", _pre_args.dataset,
            "--seed",    str(_pre_args.seed),
            "--device",  _pre_args.device]

# ── Step 3: import RLMRec modules (cwd must be external/RLMRec/) ─────────────
sys.path.insert(0, str(RLMREC_ENCODER))

from config.configurator import configs
from models.bulid_model import build_model
from data_utils.build_data_handler import build_data_handler
from trainer.trainer import init_seed


# ── Adapter ──────────────────────────────────────────────────────────────────
class RLMRecAdapter:
    """
    Wraps RLMRec model to expose predict() compatible with benchmark's
    evaluate_model(). Pre-computes embeddings once; evaluate.py handles
    train item masking.
    """
    def __init__(self, model):
        self.model = model
        self._user_embeds = None
        self._item_embeds = None

    def eval(self):
        self.model.eval()
        self.model.is_training = False
        with torch.no_grad():
            self._user_embeds, self._item_embeds = self.model.forward(self.model.adj, 1.0)

    def predict(self, user_tensor):
        with torch.no_grad():
            return self._user_embeds[user_tensor] @ self._item_embeds.T


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = _pre_args
    init_seed()

    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs["device"])

    ckpt_path = Path(f"./encoder/checkpoint/{args.model}-{args.dataset}-{args.seed}/best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun training first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=configs["device"], weights_only=False))
    print(f"Loaded: {ckpt_path}")

    adapter = RLMRecAdapter(model)
    adapter.eval()

    interaction_data = InteractionData()

    metrics = evaluate_model(
        adapter,
        interaction_data,
        device=configs["device"],
        split=args.split,
    )

    print(f"\n=== {args.split.upper()} metrics (benchmark-standard) ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results_dir = BENCHMARK_ROOT / "results" / f"rlmrec_{args.model}__{args.dataset}__seed{args.seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"

    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    existing[f"{args.split}_metrics"] = metrics
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {results_path}")


if __name__ == "__main__":
    main()
