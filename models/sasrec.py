"""
SASRec (Kang & McAuley, ICDM 2018).
Self-Attentive Sequential Recommendation.

Models user interaction sequences with a causal self-attention transformer.
Side features can be injected by adding content embeddings to item embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Set, List


class SASRec(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        max_seq_len: int = 50,
        n_heads: int = 2,
        n_blocks: int = 2,
        dropout: float = 0.2,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Item embedding + positional embedding
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)  # 0 = padding
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Optional feature projection
        self.feature_proj = None
        if feature_dim > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.item_features = None

    def set_features(self, item_features: torch.Tensor):
        """Set item content features. Shape: (n_items, feature_dim)."""
        # Pad with zeros for the padding item (index 0)
        padding = torch.zeros(1, item_features.shape[1], device=item_features.device)
        self.item_features = torch.cat([padding, item_features], dim=0)

    def _get_item_emb(self, item_ids):
        """Get item embeddings, optionally augmented with content features."""
        emb = self.item_emb(item_ids)
        if self.feature_proj is not None and self.item_features is not None:
            feat = self.feature_proj(self.item_features[item_ids])
            emb = emb + feat
        return emb

    def forward_seq(self, seq):
        """
        Forward pass on item sequences.
        seq: (batch, seq_len) item IDs (0-padded)
        Returns: (batch, seq_len, embed_dim) contextualized representations
        """
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        item_embs = self._get_item_emb(seq)
        x = item_embs + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Causal mask: prevent attending to future items
        mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # Padding mask
        pad_mask = (seq == 0)

        for block in self.blocks:
            x = block(x, mask, pad_mask)

        x = self.norm(x)
        return x

    def forward(self, users, pos_items, neg_items, sequences=None):
        """
        BPR-style forward for sequence model.
        sequences: (batch, seq_len) — user interaction sequences
        """
        if sequences is None:
            raise ValueError("SASRec requires sequences input")

        seq_out = self.forward_seq(sequences)
        # Use last position's output as user representation
        # Find the last non-padding position for each sequence
        seq_lens = (sequences != 0).sum(dim=1) - 1  # (batch,)
        seq_lens = seq_lens.clamp(min=0)
        user_repr = seq_out[torch.arange(len(seq_lens)), seq_lens]  # (batch, dim)

        pos_emb = self._get_item_emb(pos_items)
        neg_emb = self._get_item_emb(neg_items)

        pos_scores = (user_repr * pos_emb).sum(dim=1)
        neg_scores = (user_repr * neg_emb).sum(dim=1)

        reg_loss = (user_repr.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(users)

        return pos_scores, neg_scores, reg_loss

    @torch.no_grad()
    def predict_from_seq(self, sequences):
        """
        Predict scores for all items given user sequences.
        Returns: (batch, n_items)
        """
        seq_out = self.forward_seq(sequences)
        seq_lens = (sequences != 0).sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        user_repr = seq_out[torch.arange(len(seq_lens)), seq_lens]

        # Score all items (exclude padding item 0)
        all_items = self._get_item_emb(
            torch.arange(1, self.n_items + 1, device=sequences.device)
        )
        scores = user_repr @ all_items.T
        return scores


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.dropout(x) + residual

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual

        return x
