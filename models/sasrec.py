"""
SASRec (Kang & McAuley, ICDM 2018).
Self-Attentive Sequential Recommendation.

Models user interaction sequences with a causal self-attention transformer.
Content features (M7) can be injected at 3 independent points controlled by
injection_mode: "input" | "ffn" | "output" | combinations | "all" | "none".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Set, List

_VALID_MODES = {
    "none", "input", "ffn", "output",
    "input+ffn", "input+output", "ffn+output", "all",
}


def _make_proj(feature_dim: int, embed_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
    )


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
        injection_mode: str = "none",
    ):
        assert injection_mode in _VALID_MODES, (
            f"injection_mode must be one of {_VALID_MODES}, got '{injection_mode}'"
        )
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.injection_mode = injection_mode

        self._inject_input  = "input"  in injection_mode or injection_mode == "all"
        self._inject_ffn    = "ffn"    in injection_mode or injection_mode == "all"
        self._inject_output = "output" in injection_mode or injection_mode == "all"

        # Item embedding + positional embedding
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Independent projection for each active injection point
        self.input_proj  = _make_proj(feature_dim, embed_dim) if (feature_dim > 0 and self._inject_input)  else None
        self.ffn_proj    = _make_proj(feature_dim, embed_dim) if (feature_dim > 0 and self._inject_ffn)    else None
        self.output_proj = _make_proj(feature_dim, embed_dim) if (feature_dim > 0 and self._inject_output) else None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.item_features   = None  # set via set_features()
        self.user_sequences  = None  # set via set_user_sequences()

    def set_features(self, item_features: torch.Tensor):
        """Set item content features. Shape: (n_items, feature_dim)."""
        padding = torch.zeros(1, item_features.shape[1], device=item_features.device)
        self.item_features = torch.cat([padding, item_features], dim=0)

    def set_user_sequences(self, user_sequences: dict):
        """
        Cache per-user interaction sequences for evaluation.
        user_sequences: {user_id: [item_id, ...]} sorted by timestamp.
        Call with train_sequences before val eval,
        and train_val_sequences before test eval.
        """
        self.user_sequences = user_sequences

    @torch.no_grad()
    def predict(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Standard evaluation interface: score all items for each user.
        Requires set_user_sequences() to be called first.
        Returns: (batch, n_items)
        """
        device = user_ids.device
        seqs = []
        for uid in user_ids.cpu().tolist():
            seq = (self.user_sequences or {}).get(uid, [])
            ctx = seq[-self.max_seq_len:]                    # most recent max_seq_len items
            pad = [0] * (self.max_seq_len - len(ctx))
            # Shift item IDs by +1 (0 reserved for padding); right-padding to match training
            seqs.append([item + 1 for item in ctx] + pad)
        seq_tensor = torch.tensor(seqs, dtype=torch.long, device=device)
        return self.predict_from_seq(seq_tensor)

    # ------------------------------------------------------------------
    # Injection ① INPUT — content added at embedding stage
    # ------------------------------------------------------------------
    def _get_item_emb(self, item_ids: torch.Tensor, stage: str = "input") -> torch.Tensor:
        """
        Get item embeddings, optionally augmented with content features.
        stage: "input"  → uses input_proj  (injection ①)
               "score"  → uses output_proj (injection ③)
        """
        emb = self.item_emb(item_ids)
        if self.item_features is None:
            return emb

        if stage == "input" and self.input_proj is not None:
            emb = emb + self.input_proj(self.item_features[item_ids])

        elif stage == "score" and self.output_proj is not None:
            emb = emb + self.output_proj(self.item_features[item_ids])

        return emb

    # ------------------------------------------------------------------
    # Injection ② FFN — content projected and passed into TransformerBlock
    # ------------------------------------------------------------------
    def forward_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on item sequences.
        seq: (batch, seq_len) item IDs (0-padded)
        Returns: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)

        x = self._get_item_emb(seq, stage="input") + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Causal mask: prevent attending to future items
        mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()
        pad_mask = (seq == 0)

        # Pre-compute FFN content embedding once for all blocks
        content_emb = None
        if self.ffn_proj is not None and self.item_features is not None:
            content_emb = self.ffn_proj(self.item_features[seq])  # (batch, seq_len, embed_dim)

        for block in self.blocks:
            x = block(x, mask, pad_mask, content_emb=content_emb)

        x = self.norm(x)
        return x

    def forward(self, users, pos_items, neg_items, sequences=None):
        """BPR-style forward for sequence model."""
        if sequences is None:
            raise ValueError("SASRec requires sequences input")

        seq_out = self.forward_seq(sequences)
        seq_lens = (sequences != 0).sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        user_repr = seq_out[torch.arange(len(seq_lens)), seq_lens]

        # Use "score" stage so injection ③ is consistent between train and eval
        pos_emb = self._get_item_emb(pos_items, stage="score")
        neg_emb = self._get_item_emb(neg_items, stage="score")

        pos_scores = (user_repr * pos_emb).sum(dim=1)
        neg_scores = (user_repr * neg_emb).sum(dim=1)

        reg_loss = (
            user_repr.norm(2).pow(2)
            + pos_emb.norm(2).pow(2)
            + neg_emb.norm(2).pow(2)
        ) / pos_items.shape[0]

        return pos_scores, neg_scores, reg_loss

    # ------------------------------------------------------------------
    # Injection ③ OUTPUT — content added at scoring step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_from_seq(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for all items given user sequences.
        Returns: (batch, n_items)
        """
        seq_out = self.forward_seq(sequences)
        seq_lens = (sequences != 0).sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        user_repr = seq_out[torch.arange(len(seq_lens)), seq_lens]

        # Score all items (exclude padding item 0); output_proj applied if mode includes "output"
        all_items = self._get_item_emb(
            torch.arange(1, self.n_items + 1, device=sequences.device),
            stage="score",
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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        content_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.dropout(x) + residual

        # Feed-forward with residual — injection ② applied before self.ff
        residual = x
        x = self.norm2(x)
        if content_emb is not None:
            x = x + content_emb
        x = self.ff(x) + residual

        return x
