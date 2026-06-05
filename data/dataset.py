"""
PyTorch Dataset for BPR training and evaluation on ML-20M.

Provides:
  - BPRDataset: yields (user, pos_item, neg_item) triplets for BPR training
  - SequenceDataset: yields (seq_padded, target_item, neg_item) for sequential models
  - InteractionData: loads train/val/test splits, builds adjacency, provides user histories
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
import scipy.sparse as sp

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, BATCH_SIZE, NUM_NEGATIVES, SEQ_BATCH_SIZE, MAX_SEQ_LEN


class InteractionData:
    """
    Loads processed train/val/test splits and provides:
      - user/item counts
      - user interaction histories (for negative sampling and evaluation)
      - sparse adjacency matrix (for GNN models)
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

        # Load splits
        self.train_df = pd.read_csv(data_dir / "train.csv")
        self.val_df = pd.read_csv(data_dir / "val.csv")
        self.test_df = pd.read_csv(data_dir / "test.csv")

        # Load stats
        with open(data_dir / "stats.json") as f:
            self.stats = json.load(f)

        self.n_users = self.stats["n_users"]
        self.n_items = self.stats["n_items"]

        # Load ID mappings
        with open(data_dir / "item_map.json") as f:
            self.item_map = {int(k): v for k, v in json.load(f).items()}
        self.item_map_inv = {v: k for k, v in self.item_map.items()}

        # Build user interaction sets (for BPR models and negative sampling)
        self.train_user_items = self._build_user_items(self.train_df)
        self.val_user_items   = self._build_user_items(self.val_df)
        self.test_user_items  = self._build_user_items(self.test_df)

        # Build per-user sorted sequences (for sequential models)
        # train_sequences: used as context during training and val evaluation
        # train_val_sequences: used as context during test evaluation
        self.train_sequences     = self._build_user_sequences(self.train_df)
        train_val_df             = pd.concat([self.train_df, self.val_df], ignore_index=True)
        self.train_val_sequences = self._build_user_sequences(train_val_df)

        # All items set for negative sampling
        self.all_items = set(range(self.n_items))

    def _build_user_items(self, df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Build {user_id: set(item_ids)} from a DataFrame."""
        user_items = {}
        for uid, group in df.groupby("userId"):
            user_items[int(uid)] = set(group["movieId"].tolist())
        return user_items

    def _build_user_sequences(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """Build {user_id: [item_ids sorted by timestamp]} from a DataFrame."""
        seqs = {}
        for uid, group in df.groupby("userId"):
            seqs[int(uid)] = group.sort_values("timestamp")["movieId"].tolist()
        return seqs

    def get_sparse_adj(self) -> sp.coo_matrix:
        """
        Build sparse bipartite adjacency matrix for LightGCN.
        Shape: (n_users + n_items, n_users + n_items)
        """
        users = self.train_df["userId"].values
        items = self.train_df["movieId"].values + self.n_users  # offset item IDs

        # Symmetric: user→item and item→user edges
        rows = np.concatenate([users, items])
        cols = np.concatenate([items, users])
        vals = np.ones(len(rows), dtype=np.float32)

        adj = sp.coo_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users + self.n_items, self.n_users + self.n_items),
        )
        return adj

    def get_norm_adj(self) -> torch.sparse.FloatTensor:
        """
        Normalized adjacency: D^{-1/2} A D^{-1/2} as sparse torch tensor.
        Used by LightGCN for message passing.
        """
        adj = self.get_sparse_adj()

        # Degree normalization
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)

        norm_adj = d_mat @ adj @ d_mat
        norm_adj = norm_adj.tocoo()

        indices = torch.LongTensor(np.stack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def get_train_pairs(self) -> np.ndarray:
        """Return (user, item) positive pairs from training data."""
        return self.train_df[["userId", "movieId"]].values


class BPRDataset(Dataset):
    """
    BPR triplet dataset: yields (user, pos_item, neg_item).
    Negative items are sampled uniformly from items the user has NOT interacted with.
    """

    def __init__(self, interaction_data: InteractionData):
        self.data = interaction_data
        self.pairs = self.data.get_train_pairs()
        self.n_items = self.data.n_items
        self.train_user_items = self.data.train_user_items

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user, pos_item = self.pairs[idx]

        # Sample negative item (not in user's training history)
        neg_item = np.random.randint(self.n_items)
        user_items = self.train_user_items.get(user, set())
        while neg_item in user_items:
            neg_item = np.random.randint(self.n_items)

        return int(user), int(pos_item), int(neg_item)


def get_train_loader(interaction_data: InteractionData, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Create training DataLoader with BPR sampling."""
    dataset = BPRDataset(interaction_data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )


class SequenceDataset(Dataset):
    """
    Dataset for sequential models (SASRec / BERT4Rec / DuoRec).

    Sliding-window over each user's chronologically-sorted training history:
      given sequence [i1, i2, ..., iN], produces N-1 samples:
        context=[i1..i_{t-1}] (left-padded to max_seq_len), target=i_t
      for t = 1 … N-1.

    Each sample is a (seq_padded, target_item, neg_item) triplet.
    """

    def __init__(
        self,
        interaction_data: InteractionData,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.max_seq_len      = max_seq_len
        self.n_items          = interaction_data.n_items
        self.train_user_items = interaction_data.train_user_items

        # Build sample index: list of (seq_ref, position_t)
        # seq_ref points into interaction_data.train_sequences — no data copy.
        self._samples: List[Tuple[List[int], int]] = []
        for seq in interaction_data.train_sequences.values():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                self._samples.append((seq, t))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        seq, t = self._samples[idx]

        # Context = everything before position t, truncated from left to max_seq_len.
        # Shift item IDs by +1: dataset uses 0-indexed (0..n_items-1) but SASRec
        # reserves 0 as the padding token, so valid items must be 1..n_items.
        ctx = seq[max(0, t - self.max_seq_len): t]
        pad_len = self.max_seq_len - len(ctx)
        # Right-padding: items first, then zeros.
        # Left-padding + causal mask causes softmax(-inf)=NaN for position 0.
        padded = [item + 1 for item in ctx] + [0] * pad_len

        target = seq[t] + 1          # 1-indexed target

        # Negative: sample 1-indexed item different from target
        neg = np.random.randint(1, self.n_items + 1)
        while neg == target:
            neg = np.random.randint(1, self.n_items + 1)

        return (
            torch.tensor(padded,  dtype=torch.long),
            torch.tensor(target,  dtype=torch.long),
            torch.tensor(neg,     dtype=torch.long),
        )


def get_seq_train_loader(
    interaction_data: InteractionData,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int  = SEQ_BATCH_SIZE,
) -> DataLoader:
    """Create training DataLoader for sequential models."""
    dataset = SequenceDataset(interaction_data, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
