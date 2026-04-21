"""
LightGCL (Cai et al., ICLR 2023).
Simple yet effective graph contrastive learning for recommendation.

Key insight: uses SVD decomposition of the adjacency matrix to create
informative contrastive views, replacing random noise perturbation.
The low-rank SVD reconstruction captures global collaborative patterns
that serve as a structurally meaningful augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LightGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        svd_q: int = 5,         # SVD rank for contrastive view
        cl_weight: float = 0.2,  # contrastive loss weight
        cl_temp: float = 0.2,    # InfoNCE temperature
        norm_adj: Optional[torch.sparse.FloatTensor] = None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.svd_q = svd_q
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)

        self.norm_adj = norm_adj
        # SVD-reconstructed adjacency (computed lazily in set_adj)
        self.svd_adj = None

    def set_adj(self, norm_adj):
        """Set adjacency and compute SVD-based contrastive adjacency."""
        self.norm_adj = norm_adj
        self._compute_svd_adj()

    def _compute_svd_adj(self):
        """Compute low-rank SVD approximation of the adjacency matrix."""
        adj = self.norm_adj.cpu()

        # Extract user→item sparse subblock directly from sparse indices (no to_dense)
        indices = adj.coalesce().indices()
        values = adj.coalesce().values()
        mask = (indices[0] < self.n_users) & (indices[1] >= self.n_users)
        row = indices[0][mask]
        col = indices[1][mask] - self.n_users
        val = values[mask]
        ui_sparse = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            val,
            (self.n_users, self.n_items),
        ).coalesce()

        # Truncated SVD on CPU (only top-q components, memory-efficient)
        U_q, S_q, V_q = torch.svd_lowrank(ui_sparse, q=self.svd_q)

        # Store low-rank factors on device — never materialize full matrix
        device = self.norm_adj.device
        self.svd_U = U_q.to(device)   # (n_users, q)
        self.svd_S = S_q.to(device)   # (q,)
        self.svd_V = V_q.to(device)   # (n_items, q)

    def _propagate(self, adj):
        """LightGCN propagation with given adjacency."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            emb_list.append(all_emb)

        final = torch.stack(emb_list, dim=0).mean(dim=0)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final

    def _propagate_svd(self):
        """LightGCN propagation using low-rank SVD factors (no full matrix)."""
        # svd_adj ≈ [[0, U S V^T], [V S U^T, 0]]
        # Multiply layer by layer without materializing full matrix
        user_emb = self.user_emb.weight   # (n_users, d)
        item_emb = self.item_emb.weight   # (n_items, d)
        user_list = [user_emb]
        item_list = [item_emb]

        for _ in range(self.n_layers):
            # user_new = U S V^T @ item_emb
            user_emb = self.svd_U @ (torch.diag(self.svd_S) @ (self.svd_V.T @ item_emb))
            # item_new = V S U^T @ user_emb (previous)
            item_emb = self.svd_V @ (torch.diag(self.svd_S) @ (self.svd_U.T @ user_list[-1]))
            user_list.append(user_emb)
            item_list.append(item_emb)

        user_final = torch.stack(user_list, dim=0).mean(dim=0)
        item_final = torch.stack(item_list, dim=0).mean(dim=0)
        return user_final, item_final

    def _infonce_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        pos = (view1 * view2).sum(dim=-1) / self.cl_temp
        neg = view1 @ view2.T / self.cl_temp
        return (-pos + torch.logsumexp(neg, dim=-1)).mean()

    def forward(self, users, pos_items, neg_items):
        # Main view: propagate on original adjacency
        user_final, item_final = self._propagate(self.norm_adj)
        # Contrastive view: propagate using low-rank SVD factors
        user_svd, item_svd = self._propagate_svd()

        u = user_final[users]
        p = item_final[pos_items]
        n = item_final[neg_items]

        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)

        # Regularization on initial embeddings
        u0 = self.user_emb(users)
        p0 = self.item_emb(pos_items)
        n0 = self.item_emb(neg_items)
        reg_loss = (u0.norm(2).pow(2) + p0.norm(2).pow(2) + n0.norm(2).pow(2)) / len(users)

        # Contrastive loss: original view vs SVD view
        cl_loss_user = self._infonce_loss(user_final[users], user_svd[users])
        cl_loss_item = self._infonce_loss(item_final[pos_items], item_svd[pos_items])
        cl_loss = (cl_loss_user + cl_loss_item) * self.cl_weight

        return pos_scores, neg_scores, reg_loss + cl_loss

    @torch.no_grad()
    def predict(self, user_ids):
        user_final, item_final = self._propagate(self.norm_adj)
        u = user_final[user_ids]
        return u @ item_final.T
