#!/usr/bin/env python3
"""
Prepare ML-20M data in RLMRec's expected format.

Converts our benchmark splits + embeddings into:
  data/ml20m/trn_mat.pkl   — sparse (n_users, n_items) train matrix
  data/ml20m/val_mat.pkl   — sparse val matrix
  data/ml20m/tst_mat.pkl   — sparse test matrix
  data/ml20m/itm_emb_np.pkl — item embeddings (n_items, dim)
  data/ml20m/usr_emb_np.pkl — user embeddings (n_users, dim)
  data/ml20m/itm_prf.pkl   — item profiles (optional)
  data/ml20m/usr_prf.pkl   — user profiles (optional)

Usage:
  python prepare_rlmrec_data.py                          # use LLM profile embeddings
  python prepare_rlmrec_data.py --item-features genome   # use genome embeddings
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix

BENCHMARK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BENCHMARK_ROOT))
from config import DATA_DIR
from features.loader import FeatureLoader


def build_sparse_matrix(csv_path: Path, n_users: int, n_items: int) -> csr_matrix:
    """Build sparse interaction matrix from CSV (userId, movieId, timestamp)."""
    df = pd.read_csv(csv_path)
    rows = df["userId"].values
    cols = df["movieId"].values
    vals = np.ones(len(df), dtype=np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


def build_user_embeddings(train_csv: Path, item_emb: np.ndarray, n_users: int) -> np.ndarray:
    """
    Build user embeddings as mean of their liked items' embeddings.
    This is a standard approach when user-side LLM profiles are unavailable.
    """
    df = pd.read_csv(train_csv)
    embed_dim = item_emb.shape[1]
    user_emb = np.zeros((n_users, embed_dim), dtype=np.float32)

    for uid, group in df.groupby("userId"):
        item_ids = group["movieId"].values
        user_emb[uid] = item_emb[item_ids].mean(axis=0)

    # Normalize
    norms = np.linalg.norm(user_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    user_emb = user_emb / norms

    return user_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-features", type=str, default="profile",
                        choices=["profile", "genome", "mood", "combined"],
                        help="Which item embeddings to use")
    args = parser.parse_args()

    # Load stats
    with open(DATA_DIR / "stats.json") as f:
        stats = json.load(f)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    print(f"Users: {n_users}, Items: {n_items}")

    # Output directory
    rlmrec_root = Path(__file__).parent / "RLMRec"
    out_dir = rlmrec_root / "data" / "ml20m"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if all required files already exist
    required_files = ["trn_mat.pkl", "val_mat.pkl", "tst_mat.pkl",
                      "itm_emb_np.pkl", "usr_emb_np.pkl", "itm_prf.pkl", "usr_prf.pkl"]
    if all((out_dir / f).exists() for f in required_files):
        print(f"All RLMRec data files already exist in {out_dir}/ — skipping preparation")
        print("  (delete files manually to force regeneration)")
        return

    # Build sparse matrices
    print("Building sparse matrices...")
    trn_mat = build_sparse_matrix(DATA_DIR / "train.csv", n_users, n_items)
    val_mat = build_sparse_matrix(DATA_DIR / "val.csv", n_users, n_items)
    tst_mat = build_sparse_matrix(DATA_DIR / "test.csv", n_users, n_items)

    print(f"  Train: {trn_mat.nnz:,} interactions")
    print(f"  Val:   {val_mat.nnz:,} interactions")
    print(f"  Test:  {tst_mat.nnz:,} interactions")

    with open(out_dir / "trn_mat.pkl", "wb") as f:
        pickle.dump(trn_mat, f)
    with open(out_dir / "val_mat.pkl", "wb") as f:
        pickle.dump(val_mat, f)
    with open(out_dir / "tst_mat.pkl", "wb") as f:
        pickle.dump(tst_mat, f)

    # Load item embeddings
    print(f"Loading item embeddings ({args.item_features})...")
    loader = FeatureLoader()
    feature_map = {
        "profile": ["profile"],
        "genome": ["genome"],
        "mood": ["mood"],
        "combined": ["profile", "mood"],
    }
    item_emb = loader.get_combined(feature_map[args.item_features])
    print(f"  Item embeddings: {item_emb.shape}")

    # Build user embeddings (mean of liked items)
    print("Building user embeddings (mean pooling of liked items)...")
    user_emb = build_user_embeddings(DATA_DIR / "train.csv", item_emb, n_users)
    print(f"  User embeddings: {user_emb.shape}")

    # Save embeddings
    with open(out_dir / "itm_emb_np.pkl", "wb") as f:
        pickle.dump(item_emb, f)
    with open(out_dir / "usr_emb_np.pkl", "wb") as f:
        pickle.dump(user_emb, f)

    # Save dummy profiles (required by some RLMRec models but not used in training)
    itm_prf = [{"profile": "", "reasoning": ""} for _ in range(n_items)]
    usr_prf = [{"profile": "", "reasoning": ""} for _ in range(n_users)]
    with open(out_dir / "itm_prf.pkl", "wb") as f:
        pickle.dump(itm_prf, f)
    with open(out_dir / "usr_prf.pkl", "wb") as f:
        pickle.dump(usr_prf, f)

    print(f"\nAll files saved to {out_dir}/")
    print("Files:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
