# Recommendation Benchmark: LLM-Generated Features for MovieLens 20M

A rigorous benchmark for evaluating LLM-generated movie embeddings as item side features in collaborative filtering. Designed for reproducibility and targeting top-tier venues (NeurIPS D&B, RecSys, KDD, WWW).

## Research Question

> Does having an LLM reason about genome tags, plot metadata, and user reviews produce a better item representation for recommendation than (a) pure collaborative filtering, (b) the raw genome tag vectors that ship with ML-20M, or (c) naive text embeddings of movie titles?

## Dataset

**MovieLens 20M** (Harper & Konstan, 2015) — 20 million ratings from 138,493 users on 27,278 movies. We restrict to the **10,381 genome-covered movies** (99% of ratings preserved), which are the movies that have both LLM-generated features and genome tag annotations.

### LLM-Generated Features (from `embedding_generator/`)

| Feature | Dims (default) | Source | Description |
|---------|---------------|--------|-------------|
| **Profile embedding** | 1024 | LLM profile text → bge-large-en-v1.5 | Semantic fingerprint of each movie's thematic essence, tone, and style |
| **Mood vector** | 10 | Direct extraction from LLM output | 10-axis continuous mood (dark/light, serious/playful, slow/fast, etc.) |
| **Key themes** | 528 | Multi-hot over filtered LLM themes | Categorical theme labels (redemption, obsession, identity, ...) |
| **Genome PCA** | 128 | Raw 1128-dim genome scores → PCA | Baseline: existing ML-20M content features |
| **BERT title** | 1024 | "Title \| Genres" → bge-large-en-v1.5 | Control: naive text embedding without LLM reasoning |

## Baseline Models

### Why These Models?

We select baselines along three axes to isolate the contribution of LLM features:

1. **Pure CF baselines** — establish what collaborative filtering achieves alone, without any content features. Includes classic (BPR-MF), graph-based (LightGCN), and contrastive learning (SimGCL, XSimGCL, LightGCL) approaches.

2. **Content-augmented CF** — the primary ablation host. LightGCN-SF injects different feature types into the same architecture, enabling controlled comparison: genome vs. BERT title vs. LLM profile vs. mood vs. themes vs. combinations.

3. **LLM-for-RecSys methods** — existing approaches that use LLM-generated knowledge in recommendation. We must compare against these to show our features are competitive with or complementary to prior work.

The selection forms a **controlled hierarchy** that answers 5 research questions (consistent with paper §4.2):

| # | Comparison | Research Question |
|---|-----------|-------------------|
| Q1 | M4 vs M1/M1c | Does LLM content improve over pure CF (including strong contrastive methods)? |
| Q2 | M4 vs M2/M2b | Does LLM-synthesized content beat genome tags, regardless of dimensionality? **(Central claim)** |
| Q3 | M4 vs M3 | Does LLM reasoning matter, or does naive BERT encoding suffice? |
| Q4 | M7 vs M4 | Do structured mood features add value beyond semantic embeddings? |
| Q5 | M4/M7 vs R1/R2 | Does simple additive injection outperform sophisticated alignment methods? |

Supporting comparisons (not primary research questions, but informative):

| Comparison | What It Tests |
|---|---|
| M0 (BPR-MF) vs M1 (LightGCN) | Does graph structure help? (sanity check) |
| M1 vs M1b/M1c/M1d | How far can pure CF go with contrastive learning? (establishes ceiling) |
| M4 vs M1d (LightGCL) | Does LLM content beat the strongest pure CF? (strongest evidence) |
| M7 vs M8 | Do categorical themes further help, or add noise? |
| M2 vs M9 | Do structured LLM features complement genome tags? |

For a **dataset paper**, the goal is not to propose a new model — it is to demonstrate that the LLM-generated features (profiles, moods, themes) are a **useful resource**. We need baselines at multiple strength levels so that improvements cannot be dismissed as "you just used a weak baseline." BPR-MF is the floor, LightGCL is the ceiling of pure-CF methods, and LightGCN-SF is the controlled testbed that keeps architecture constant while varying only the features.

### Tier 1: Pure Collaborative Filtering

#### BPR-MF (Rendle et al., UAI 2009)

**Algorithm.** Each user *u* and item *i* gets a learned embedding vector (**e**_u, **e**_i ∈ ℝ^128). The predicted preference score is a dot product:

```
score(u, i) = eᵤ · eᵢ
```

Training uses **Bayesian Personalized Ranking**: for each observed interaction (u, i⁺), sample an unobserved item i⁻, then maximize the margin between observed and unobserved items:

```
L_BPR = −log σ(score(u, i⁺) − score(u, i⁻))  +  λ‖embeddings‖²
```

This learns to rank observed items above unobserved ones — no pointwise rating prediction, just pairwise ordering. No content features, no graph structure — purely learned from interaction patterns.

**Why included.** The simplest CF baseline. Isolates the value of collaborative signal alone with no graph structure and no content features. If LightGCN beats BPR-MF, the graph structure helps. If LightGCN-SF beats both, the side features help *on top of* graph structure. Also serves as a sanity check that the evaluation pipeline is correct (expected NDCG@10 ~0.05-0.08 on ML-20M implicit).

#### LightGCN (He et al., SIGIR 2020)

**Algorithm.** Constructs a bipartite user-item interaction graph and computes a **normalized adjacency matrix** Â. User/item embeddings are iteratively propagated through *L* = 3 layers of graph convolution:

```
e⁽ˡ⁾ = Â · e⁽ˡ⁻¹⁾       (neighborhood averaging — no weight matrices, no nonlinearities)
```

The key simplification over standard GCNs: **no learned weight matrices, no nonlinear activations** at each layer. The final representation is the **mean pool** across all layers (including layer 0):

```
e_final = (1/(L+1)) · Σₗ₌₀ᴸ e⁽ˡ⁾
```

This captures multi-hop collaborative signals: layer 1 aggregates direct neighbors (users who rated the same item), layer 2 captures 2-hop patterns (items rated by similar users), layer 3 extends to 3-hop neighborhoods. BPR loss is applied on the final embeddings, with L2 regularization on the **initial** (non-propagated) embeddings only.

**Why included.** The standard GNN CF baseline expected at all top venues. Consistently strong on ML-20M. Our LightGCN-SF extends this architecture with side features, making LightGCN the natural ID-only comparison point.

#### SimGCL (Yu et al., SIGIR 2022)

**Algorithm.** Builds on LightGCN by adding **contrastive self-supervised learning** as regularization. The key insight: you don't need expensive graph augmentation (edge dropout, node dropout) — just adding **random noise** to embeddings creates effective contrastive views.

Two forward passes per batch:
1. **Clean view**: standard LightGCN propagation → final mean-pooled embeddings
2. **Perturbed view**: same propagation, but after each layer, add noise:
   ```
   e' = e + ε · normalize(rand_uniform)     where ε = 0.1
   ```

Then apply **InfoNCE contrastive loss** between the two views at the final layer (same-layer contrast):

```
L_CL = −sim(clean_u, pert_u)/τ  +  log Σ exp(sim(clean_u, pert_v)/τ)
```

Total loss combines BPR ranking with contrastive regularization:

```
L = L_BPR  +  λ · L_reg  +  0.2 · (L_CL_user + L_CL_item)
```

The contrastive term forces the model to learn representations that are **invariant to small perturbations** — this acts as implicit data augmentation and improves embedding uniformity, reducing representation collapse.

**Why included.** Represents the **contrastive CF paradigm**. Tests whether self-supervised contrastive learning (without content features) can match or exceed content-augmented LightGCN-SF. If M4 (LightGCN-SF + LLM profiles) beats M1b (SimGCL), it means LLM-derived content provides signal that contrastive regularization alone cannot capture.

#### XSimGCL (Yu et al., TKDE 2023)

**Algorithm.** Refines SimGCL by using **cross-layer** contrast instead of same-layer. The perturbed view still adds noise at each layer, but instead of comparing clean-final vs. perturbed-final, it compares embeddings from **different propagation depths**:

- **View 1**: clean final mean-pooled embedding (average of all layers 0 through *L*)
- **View 2**: perturbed embedding from **layer 2 specifically** (an intermediate depth)

```
L_CL = InfoNCE(final_clean[batch], layer2_perturbed[batch])
```

This forces alignment between the global multi-scale representation (mean of all layers) and a specific intermediate-depth representation, encouraging **cross-scale consistency**. The authors showed this outperforms same-layer SimGCL because it creates a harder, more informative contrastive task — the two views capture different structural granularities.

**Why included.** A strong contrastive CF baseline. Cross-layer contrast creates a harder, more informative contrastive task than same-layer SimGCL.

#### LightGCL (Cai et al., ICLR 2023)

**Algorithm.** Replaces the random noise perturbation used by SimGCL/XSimGCL with **SVD-based augmentation** of the adjacency matrix. Computes a truncated SVD (rank *q* = 5) of the user-item interaction matrix to create a low-rank approximation, then uses this as the contrastive view:

1. **Main view**: standard LightGCN propagation on the original adjacency
2. **Contrastive view**: LightGCN propagation on the SVD-reconstructed adjacency

```
A_svd = U_q · S_q · V_q^T     (rank-q approximation of user-item block)
L = L_BPR + λ · (InfoNCE(main_user, svd_user) + InfoNCE(main_item, svd_item))
```

The SVD reconstruction captures global collaborative patterns (the most important latent factors of the interaction matrix), providing a structurally meaningful augmentation rather than random perturbation. This produces more informative contrastive views that better preserve the graph's collaborative structure.

**Why included.** The **strongest pure-CF baseline** — represents the current frontier of graph contrastive learning. LightGCL's SVD-based augmentation often outperforms noise-based methods (SimGCL, XSimGCL) on standard benchmarks. If LLM features (M4, M7) can improve upon LightGCL (M1d), it provides strong evidence that content features add **orthogonal information** that even sophisticated self-supervised methods cannot recover from interaction data alone. This is the most convincing argument that LLM-generated features are genuinely useful as a dataset resource.

### Tier 2: Content-Augmented CF (Ablation Host)

#### LightGCN-SF (LightGCN with Side Features)

**Algorithm.** Extends LightGCN by injecting pre-computed content features into item embeddings **before** graph propagation, so content information flows through the graph alongside collaborative signals.

**Full forward pass (example: M4 with 1024-dim bge-large profile embeddings):**

```
Constructor: LightGCNSF(n_users=127371, n_items=9906, embed_dim=128, n_layers=3, feature_dim=1024)

Learnable parameters:
  user_emb:     Embedding(127371, 128)               — learned from interactions
  item_emb:     Embedding(9906, 128)                  — learned from interactions
  feature_proj: Linear(1024→128) → ReLU → Linear(128→128)  — learned projection

Step 1: Project content features into embedding space
  content_features (9906, 1024)
      → Linear(1024 → 128)     # compress to embedding space
      → ReLU                    # nonlinear transformation
      → Linear(128 → 128)      # refine
      = projected (9906, 128)

Step 2: Additive injection
  item_repr = item_emb(128) + projected(128)
  = (9906, 128)

Step 3: LightGCN graph propagation (3 layers)
  all_emb = [user_emb(127371, 128) ; item_repr(9906, 128)]   # concat users + items
  = (137277, 128)

  Layer 0: all_emb              (initial embeddings)
  Layer 1: Â · all_emb          (1-hop neighborhood averaging)
  Layer 2: Â · layer1           (2-hop patterns)
  Layer 3: Â · layer2           (3-hop patterns)

  final = mean(layer0, layer1, layer2, layer3)   # (137277, 128)

Step 4: BPR scoring
  user_final = final[:127371]    # (127371, 128)
  item_final = final[127371:]    # (9906, 128)
  score(u, i) = user_final[u] · item_final[i]    # dot product
```

The MLP compresses arbitrary-dimensional features (10-dim mood, 128-dim genome, 1024-dim bge-large) into the fixed 128-dim embedding space. Everything after Step 1 is identical regardless of input feature dimensionality — the graph propagation always operates in 128-dim space.

**Why this injection strategy:**
- **Addition** (not concatenation) keeps the item embedding dimension fixed at *d*=128, avoiding dimensionality confounds across experiments. A 1024-dim input and a 10-dim input both produce the same 128-dim output after the MLP.
- The **2-layer MLP with ReLU** allows nonlinear transformation — a single linear layer would be a linear projection that may not capture complex relationships between feature dimensions.
- Features are injected **before propagation** (not after) so content information can propagate to neighbors — a user's representation benefits from the content features of all items they interacted with. After 3 layers, content signal has reached 3-hop neighborhoods.

**Why included.** The **primary ablation host** and the most critical model in this benchmark. By varying only the content features while keeping the model architecture identical, we isolate the effect of each feature type. This is the core experimental design that makes the dataset paper's contribution clear.

### Tier 3: LLM-for-RecSys Methods

#### RLMRec-plus (Wei et al., WWW 2024)

**Algorithm.** Bridges ID-based collaborative filtering and LLM semantic understanding through **cross-view contrastive alignment**. Generates user/item text profiles via an LLM (we use our Claude Haiku profiles), encodes them into semantic embeddings, then applies a contrastive knowledge distillation loss that aligns the CF embedding space with the semantic embedding space — pulling the CF representation of an item toward its LLM semantic representation, and pushing it away from other items' representations. The CF backbone (LightGCN) learns to incorporate semantic structure without modifying its architecture.

**Why included.** The leading model-agnostic framework for injecting LLM knowledge into recommendation. Directly comparable to our approach — both use LLM-generated embeddings, but RLMRec uses contrastive alignment while we use additive feature injection.

#### RLMRec-gene (Wei et al., WWW 2024)

**Algorithm.** Instead of aligning CF and semantic spaces, learns to **reconstruct** LLM semantic embeddings from masked CF embeddings. Randomly masks portions of the GCN-propagated embeddings, then trains an auxiliary decoder to reconstruct the LLM semantic embedding from the masked CF embedding. This generative self-supervised objective forces the CF embeddings to encode information present in the LLM representations, without requiring explicit alignment.

**Why included.** Provides a generative counterpart to the contrastive RLMRec-plus. If RLMRec-gene outperforms our simpler additive injection, it suggests that more sophisticated fusion mechanisms are needed. If our approach matches it, it validates that simple injection of high-quality features is sufficient.

#### KAR (Xi et al., KDD 2024)

**Algorithm.** Knowledge-Augmented Recommendation integrates LLM-generated knowledge features into a LightGCN backbone via a **hybrid-expert adapter**. Instead of a single MLP projection (as in LightGCN-SF), KAR uses a mixture-of-experts (MoE) module:

1. **Multiple expert MLPs** (4 experts) each project the knowledge features into the embedding space
2. A **gating network** conditioned on the item's CF embedding produces softmax weights over experts
3. The final knowledge representation is the weighted sum of expert outputs, added to the learned embedding

```
gate_weights = softmax(W_gate · e_i^cf)           # (n_experts,)
knowledge_emb = Σ_k gate_weights[k] · Expert_k(features)
item_repr = e_i^cf + knowledge_emb
```

This allows the model to **adaptively select** how to integrate knowledge features based on each item's collaborative profile — items with rich interaction history may weight experts differently than cold-start items.

**Why included.** The leading knowledge-augmented LLM-for-RecSys method from a top venue (KDD 2024). Uses a more sophisticated feature integration mechanism (MoE adapter) than our simple additive injection. Comparison against KAR tests whether the hybrid-expert adapter provides meaningful benefit over LightGCN-SF's simpler MLP projection when using the same LLM-generated features.

## Experimental Settings

### Data Preprocessing

| Step | Result | Rationale |
|------|--------|-----------|
| Genome filter | 20M → 19.8M ratings, 10,370 items | Only genome-covered movies have LLM features |
| Implicit conversion (rating ≥ 3.5) | 19.8M → 12.1M interactions | Ratings below 3.5 are not positive signals — a user rating 1.5 doesn't mean "recommend this." We keep only positive interactions and sample negatives during training. |
| 10-core filtering | 12.1M → 12.05M, 127,371 users, 9,906 items | Remove users and items with fewer than 10 interactions iteratively until convergence. Standard practice to ensure sufficient collaborative signal. |
| Temporal split | Train: 11.5M (99.0%) / Val: 50K (0.4%) / Test: 67K (0.6%) | Split by timestamp, not randomly (see below). |

### Why Temporal Split (Not Random)

Random splits leak future information into training. If a user rates Movie C in 2014 (influenced by having seen Movies A and B in 2012), a random split might place Movie C in training and Movie A in test — the model trains on a preference shaped by experiences it hasn't "seen."

Temporal split prevents this by enforcing chronological order:

```
Train: all interactions before 2014-01-01     (99.0%,  11,499,778)
Val:   interactions 2014-01-01 to 2014-07-01  ( 0.4%,      49,668)
Test:  interactions after 2014-07-01          ( 0.6%,      67,466)
```

This mirrors real deployment (train on history, recommend for the future) and is required by all top venues since Rendle et al. (RecSys 2019) demonstrated that random splits inflate metrics by 10-30%.

The heavily skewed split reflects two factors: (1) ML-20M ratings are concentrated in earlier years (most activity is pre-2014), and (2) after remapping, val/test users must also appear in train, which further reduces val/test size. This is standard for temporal splits on ML-20M — the val/test sets still contain 50K/67K interactions, which is sufficient for reliable metric estimation across 127K users.

### Evaluation Protocol: Full Ranking

For each test user:
1. Compute predicted scores for **all 9,906 items**
2. Mask items the user interacted with during training (no re-recommendation)
3. Rank all remaining items by predicted score
4. Compute metrics on the full ranking

**Why full ranking, not sampled negatives?** Krichene & Rendle (KDD 2020) proved that sampled evaluation (e.g., scoring against 100 random negatives) produces inconsistent model orderings — Model A can beat Model B under sampling but lose under full ranking. The "winner" depends on the sample, not the model quality. With only 9,906 items, full ranking is computationally trivial (one matrix multiply per user batch).

### Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **NDCG@K** | Normalized Discounted Cumulative Gain | Ranking quality, position-aware. Higher-ranked relevant items contribute more. **Headline metric.** |
| **Recall@K** | \|relevant ∩ top-K\| / \|relevant\| | What fraction of relevant items appear in top-K. Measures coverage of user interests. |
| **HR@K** | 1 if any relevant item in top-K, else 0 | Binary: did the model surface anything useful in top-K? Most lenient metric. |
| **MRR** | 1 / rank_of_first_relevant_item | How quickly the model finds the first relevant item. Sensitive to top-1 performance. |

Reported at K = {10, 20, 50}. All metrics averaged across test users.

### Statistical Rigor

- **5 random seeds** (42, 123, 456, 789, 2026) controlling parameter initialization and negative sampling
- Report **mean ± standard deviation** for all metrics
- **Paired t-test** (or Wilcoxon signed-rank if non-normal) for key comparisons
- Significance threshold: p < 0.05
- Each model variant tuned independently on the validation set

### Cold-Start Analysis

Content features are most valuable when collaborative signals are sparse. We report metrics separately by item training interaction count:

| Bucket | Items | Train Interactions |
|--------|-------|--------------------|
| Cold | 112 | < 10 |
| Medium | 1,872 | 10 - 50 |
| Warm | 7,922 | > 50 |

We expect LLM features to show the largest lift on cold and medium items, where pure CF has insufficient signal.

## Complete Model Reference

Every model configuration in this benchmark, with full architectural details.

### Tier 1: Pure Collaborative Filtering (no content features)

These models use **only** interaction data (user-item pairs). No `.npy` embedding files are loaded.

| ID | Model | Architecture | Graph? | Contrastive? | Embed Dim | Key Mechanism |
|----|-------|-------------|--------|-------------|-----------|---------------|
| M0 | BPR-MF | Matrix factorization | No | No | 128 | `score = user_emb · item_emb`, BPR pairwise loss |
| M1 | LightGCN | Graph convolution | Yes (3 layers) | No | 128 | `final = mean(layer0..layer3)`, neighborhood averaging |
| M1b | SimGCL | LightGCN + noise | Yes (3 layers) | Yes (same-layer) | 128 | InfoNCE(clean_final, noisy_final), ε=0.1 |
| M1c | XSimGCL | LightGCN + noise | Yes (3 layers) | Yes (cross-layer) | 128 | InfoNCE(clean_final, noisy_layer2), ε=0.1 |
| M1d | LightGCL | LightGCN + SVD | Yes (3 layers) | Yes (SVD view) | 128 | InfoNCE(original_adj_view, svd_adj_view), rank q=5 |

**No content features. No MLP. No `.npy` files loaded.** These establish what CF alone can achieve.

### Tier 2: LightGCN-SF — Content-Augmented (primary ablation)

All Tier 2 models use the **identical LightGCN-SF architecture**. Only the input features change.

```
Architecture (same for all):
  item_repr = item_emb(128) + MLP(features → 128)
  MLP = Linear(feature_dim → 128) → ReLU → Linear(128 → 128)
  Then: 3 layers of LightGCN graph propagation on the augmented embeddings
```

| ID | Side Features | Feature Source | Input Dim | After MLP | Purpose |
|----|--------------|----------------|-----------|-----------|---------|
| M2 | Genome PCA | `genome_embeddings.npy` | 128 | 128 | Existing content baseline |
| M3 | BERT title+genre | `bert_title_embeddings.npy` | 1024 | 128 | Naive text encoding control |
| **M4** | **LLM profile** | **`profile_embeddings.npy`** | **1024** | **128** | **Core contribution** |
| M5 | LLM mood | `mood_vectors.npy` | 10 | 128 | Structured signal only |
| M6 | LLM themes | `theme_matrix.npy` | 528 | 128 | Categorical themes only |
| **M7** | **LLM profile + mood** | **profile + mood concatenated** | **1034** | **128** | **Recommended combo** |
| M8 | LLM all | profile + mood + themes | 1562 | 128 | Full LLM feature set |
| M9 | Genome + mood + themes | genome + mood + themes | 666 | 128 | Genome + structured LLM |

Default dimensions shown for bge-large-en-v1.5 (1024-dim, no PCA). With `--pca-dims 128`: M3/M4=128, M7=138, M8=666. Dimensions are read dynamically from `.npy` files — no code changes needed.

**Key point:** Regardless of input feature dimensionality (10, 128, 528, 1024, 1562), the MLP always outputs 128-dim, which is added to the 128-dim learned embedding. The graph propagation always operates in 128-dim space. This makes all Tier 2 experiments directly comparable — the only variable is feature quality, not architecture.

### Tier 3: LLM-for-RecSys Methods

These use more sophisticated integration mechanisms than LightGCN-SF's simple additive injection.

| ID | Model | Backbone | Integration Method | Features | Key Difference from LightGCN-SF |
|----|-------|----------|-------------------|----------|-------------------------------|
| R1 | RLMRec-plus | LightGCN | Contrastive distillation | LLM embeddings | Aligns CF space with LLM space via InfoNCE, doesn't inject features directly |
| R2 | RLMRec-gene | LightGCN | Generative reconstruction | LLM embeddings | Reconstructs LLM embeddings from masked CF embeddings |
| R3 | KAR | LightGCN | MoE adapter (4 experts) | LLM profile + mood | 4 expert MLPs + gating network instead of 1 MLP; gate is conditioned on CF embedding |

**The question Tier 3 answers:** Does a more sophisticated integration mechanism (contrastive alignment, generative reconstruction, mixture-of-experts) outperform LightGCN-SF's simple `learned_emb + MLP(features)`? If not, **feature quality matters more than integration sophistication** — the strongest argument for releasing the dataset.

### Key Comparisons

The 5 primary research questions (matches paper §4.2 and the hierarchy table above):

| # | Comparison | Research Question |
|---|-----------|-------------------|
| Q1 | M4 vs M1/M1c | Does LLM content improve over pure CF (including strong contrastive methods)? |
| Q2 | M4 vs M2/M2b | Does LLM-synthesized content beat genome tags, regardless of dimensionality? **(Central claim)** |
| Q3 | M4 vs M3 | Does LLM reasoning matter, or does naive BERT encoding suffice? |
| Q4 | M7 vs M4 | Do structured mood features add value beyond semantic embeddings? |
| Q5 | M4/M7 vs R1/R2 | Does simple additive injection outperform sophisticated alignment methods? |

Supporting comparisons (not primary research questions, but informative):

| Comparison | What It Tests |
|---|---|
| M0 (BPR-MF) vs M1 (LightGCN) | Does graph structure help? (sanity check) |
| M1 vs M1b/M1c/M1d | How far can pure CF go with contrastive learning? (establishes ceiling) |
| M4 vs M1d (LightGCL) | Does LLM content beat the strongest pure CF? (strongest evidence) |
| M7 vs M8 | Do categorical themes further help, or add noise? |
| M2 vs M9 | Do structured LLM features complement genome tags? |

## Experiment Status (as of 2026-04-07)

### Completed (2 / 80)

Each ran with lr=1e-3, weight_decay=1e-5, patience=20.

| Config | Model | Features | Seed | Best Epoch | Test NDCG@10 | Test HR@10 | Test MRR |
|--------|-------|----------|------|-----------|-------------|-----------|---------|
| **M0** | BPR-MF | none | 42 | 10 | 0.1083 | 0.4241 | 0.2254 |
| **M0** | BPR-MF | none | 123 | 10 | 0.1088 | 0.4173 | 0.2259 |

### Not Started (78 / 80)

- All remaining seed variants (456, 789, 2026) for M0
- All 5 seeds for: M1 (LightGCN), M1b (SimGCL), M1c (XSimGCL), M1d (LightGCL), M2 (genome), M3 (BERT), M4 (LLM profile), M5 (LLM mood), M6 (themes), M7 (profile+mood), M8 (all LLM), M9 (genome+LLM)
- RLMRec: 0 / 10 (R1 × 5 seeds, R2 × 5 seeds)
- KAR: 0 / 5 (R3 × 5 seeds)

### Early Observations

- Both completed BPR-MF experiments converge quickly (10 epochs, well under the 200 max)
- Full 5-seed runs and content-augmented models needed before drawing any conclusions

## How to Run

### Prerequisites

```bash
# 1. Data preprocessing (already done — generates train/val/test splits)
python data/preprocess.py

# 2. Generate BERT title baseline (if not already done)
python features/bert_baseline.py
```

### Single Experiment

```bash
# Basic usage (uses default paths from config.py / env vars)
python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42

# With custom paths via CLI arguments
python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
    --embedding-dir /path/to/embeddings \
    --data-dir /path/to/processed/data \
    --results-dir /path/to/results \
    --checkpoint-dir /path/to/checkpoints

# Override training hyperparameters
python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
    --lr 5e-4 --wd 1e-4 --epochs 100 --device cuda
```

**All `run_experiment.py` arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model: `bpr_mf`, `lightgcn`, `lightgcn_sf`, `simgcl`, `xsimgcl`, `lightgcl`, `kar` |
| `--features` | `none` | Feature config: `none`, `genome`, `bert_title`, `llm_profile`, `llm_mood`, `llm_themes`, `llm_prof_mood`, `llm_all`, `genome_llm` |
| `--seed` | `42` | Random seed |
| `--lr` | `1e-3` | Learning rate |
| `--wd` | `1e-5` | Weight decay (L2 regularization) |
| `--epochs` | `200` | Max training epochs |
| `--device` | `auto` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `--embedding-dir` | from config | Path to embedding `.npy` files |
| `--data-dir` | from config | Path to processed train/val/test CSVs |
| `--results-dir` | from config | Where to save experiment results |
| `--checkpoint-dir` | from config | Where to save model checkpoints |

### Quick Ablation (BPR-MF + all LightGCN-SF configs, 1 seed)

```bash
# Default (10 epochs, seed 42)
bash run_quick.sh

# Custom epochs and seed
EPOCHS=50 SEED=123 bash run_quick.sh

# Custom paths via env vars
EMBEDDING_DIR=/path/to/embeddings \
RESULTS_DIR=/path/to/results \
CHECKPOINT_DIR=/path/to/checkpoints \
    bash run_quick.sh
```

### Full Ablation (all models × all features × 5 seeds)

```bash
# Run all experiments
python run_ablation.py

# Filter to a specific model
python run_ablation.py --model lightgcn_sf

# Quick test (1 seed, 20 epochs)
python run_ablation.py --quick

# With custom paths
python run_ablation.py \
    --embedding-dir /path/to/embeddings \
    --results-dir /path/to/results \
    --checkpoint-dir /path/to/checkpoints

# Collect and print results summary (no training)
python run_ablation.py --collect-only
python run_ablation.py --collect-only --results-dir /path/to/results
```

### Tier 3: RLMRec Experiments

```bash
cd external && python prepare_rlmrec_data.py && bash run_rlmrec.sh
```

### Path Configuration Priority

Paths are resolved in this order (highest priority first):

1. **CLI arguments** (`--embedding-dir`, `--results-dir`, etc.)
2. **Environment variables** (`EMBEDDING_DIR`, `RESULTS_DIR`, etc.)
3. **Defaults** from `config.py` (relative to project root)

| Path | CLI Arg | Env Var | Default |
|------|---------|---------|---------|
| Embeddings | `--embedding-dir` | `EMBEDDING_DIR` | `../embedding_generator/output/bge-large-v1.5` |
| Processed data | `--data-dir` | `DATA_DIR` | `data/processed/` |
| Results | `--results-dir` | `RESULTS_DIR` | `results/` |
| Checkpoints | `--checkpoint-dir` | `CHECKPOINT_DIR` | `checkpoints/` |

## Running with Different Embeddings

The benchmark is **dimension-agnostic**: all feature dimensions are read dynamically from `.npy` file shapes at runtime. You can swap in embeddings from any encoder without changing benchmark code.

### How It Works

Override the embedding directory via `--embedding-dir` CLI argument or the `EMBEDDING_DIR` environment variable. By default it points to `../embedding_generator/output/bge-large-v1.5` (1024-dim, no PCA).

The pipeline:

```
--embedding-dir /path/to/embeddings  (or EMBEDDING_DIR env var)
    → features/loader.py reads .npy files, infers dimensions from file shapes
    → run_experiment.py gets feature_dim dynamically via loader.get_feature_dim()
    → model MLP auto-adapts: Linear(feature_dim → 128) regardless of input size
```

### Available Embedding Sets

Currently generated embeddings live under `../embedding_generator/output/`:

```
embedding_generator/output/
├── profile_embeddings.npy          # (10381, 1024) — bge-large-en-v1.5, no PCA (default)
├── mood_vectors.npy                # (10381, 10)   — always 10-dim
├── theme_matrix.npy                # (10381, 528)  — always 528-dim
├── genome_embeddings.npy           # (10381, 128)  — genome PCA-128
├── bert_title_embeddings.npy       # (10381, 1024) — BERT title+genre, bge-large
├── combined_features.npy           # (10381, 1034) — profile(1024) + mood(10)
├── combined_full.npy               # (10381, 1562) — profile(1024) + mood(10) + themes(528)
├── movie_id_index.json             # row-to-movieId mapping (shared across all sets)
├── theme_vocabulary.json
├── embedding_metadata.json
│
└── minilm-pca128/                  # Example: lightweight encoder with PCA
    ├── profile_embeddings.npy      # (10381, 128)
    ├── mood_vectors.npy            # (10381, 10)
    ├── theme_matrix.npy            # (10381, 528)
    ├── genome_embeddings.npy       # (10381, 128)
    ├── combined_features.npy       # (10381, 138)
    ├── combined_full.npy           # (10381, 666)
    ├── movie_id_index.json
    ├── theme_vocabulary.json
    └── embedding_metadata.json
```

### Step-by-Step: Run Experiments with a Different Encoder

#### 1. Generate new embeddings

From the `embedding_generator/` directory:

```bash
cd ../embedding_generator

# Default: bge-large-en-v1.5, no PCA (1024-dim) — outputs to output/
python main.py

# With PCA reduction
python main.py --pca-dims 128 --output-dir output/bge-large-pca128

# PCA sweep: generates full-dim + 128/256/512 variants in subdirectories
python main.py --pca-sweep 128 256 512

# Different encoder
python main.py --model Alibaba-NLP/gte-large-en-v1.5 \
    --output-dir output/gte-large-v1.5

# Lightweight encoder with PCA
python main.py --model all-MiniLM-L6-v2 --pca-dims 128 \
    --output-dir output/minilm-pca128
```

Each invocation creates a self-contained directory with all required `.npy` files, `movie_id_index.json`, and `embedding_metadata.json`.

PCA sweep (`--pca-sweep`) creates subdirectories for each dimension:

```
output/gte-large-v1.5/
├── profile_embeddings.npy      # full 1024-dim (no PCA)
├── ...                         # other files at full dim
├── pca128/
│   ├── profile_embeddings.npy  # (10381, 128)
│   ├── combined_features.npy   # (10381, 138)
│   └── ...
├── pca256/
│   └── ...
└── pca512/
    └── ...
```

#### 2. Generate BERT title baseline for the new encoder (optional)

The BERT title baseline uses the same encoder as profile embeddings. If you want a matched comparison, regenerate it:

```bash
cd ../benchmark
# The default bert_baseline.py uses bge-large-en-v1.5 (matching the profile encoder).
# For a different encoder, edit features/bert_baseline.py or copy the
# generated bert_title_embeddings.npy into your new output directory.
python features/bert_baseline.py
```

If your new embedding directory does not contain `bert_title_embeddings.npy`, experiments using `--features bert_title` (M3) will be skipped automatically by `run_ablation.py`.

#### 3. Run benchmark experiments

```bash
cd ../benchmark

# Default embeddings (bge-large, no PCA) — no extra args needed
python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42

# Use a different embedding set via CLI arg
python run_experiment.py --model lightgcn_sf --features llm_profile --seed 42 \
    --embedding-dir ../embedding_generator/output/bge-large-pca128

# Full ablation with a different encoder
python run_ablation.py --embedding-dir ../embedding_generator/output/gte-large-v1.5

# Same thing via env var
EMBEDDING_DIR=../embedding_generator/output/gte-large-v1.5 python run_ablation.py

# Quick test (1 seed, 20 epochs)
python run_ablation.py --quick --embedding-dir ../embedding_generator/output/gte-large-v1.5
```

#### 4. Separate results per encoder

Use `--results-dir` and `--checkpoint-dir` to keep results from different encoders isolated — no need to rename or move directories:

```bash
# Run with bge-large embeddings, save to encoder-specific dirs
python run_ablation.py \
    --embedding-dir ../embedding_generator/output/bge-large-v1.5 \
    --results-dir results-bge-large \
    --checkpoint-dir checkpoints-bge-large

# Run with gte-large embeddings, save to separate dirs
python run_ablation.py \
    --embedding-dir ../embedding_generator/output/gte-large-v1.5 \
    --results-dir results-gte-large \
    --checkpoint-dir checkpoints-gte-large

# Collect results from each encoder independently
python run_ablation.py --collect-only --results-dir results-bge-large
python run_ablation.py --collect-only --results-dir results-gte-large
```

If you prefer env vars (e.g., for shell scripts):

```bash
EMBEDDING_DIR=../embedding_generator/output/gte-large-v1.5 \
RESULTS_DIR=results-gte-large \
CHECKPOINT_DIR=checkpoints-gte-large \
    bash run_quick.sh
```

### Required Files per Embedding Directory

The benchmark expects these files in `EMBEDDING_DIR`. Missing optional files cause specific feature configs to be unavailable, not a crash.

| File | Required by | Notes |
|------|------------|-------|
| `profile_embeddings.npy` | M4, M7, M8 | Any dimension; shape `(10381, D)` |
| `mood_vectors.npy` | M5, M7, M8, M9 | Always `(10381, 10)` |
| `theme_matrix.npy` | M6, M8, M9 | Always `(10381, 528)` |
| `genome_embeddings.npy` | M2, M9 | PCA-128 from genome-scores; encoder-independent |
| `bert_title_embeddings.npy` | M3 | Optional; skipped if missing |
| `movie_id_index.json` | All | Row-to-movieId mapping; must match `.npy` row order |
| `combined_features.npy` | — | Not used by benchmark (convenience file) |
| `combined_full.npy` | — | Not used by benchmark (convenience file) |
| `theme_vocabulary.json` | — | Not used by benchmark (reference only) |
| `embedding_metadata.json` | — | Not used by benchmark (reference only) |

The `combined_*.npy` files are pre-concatenated convenience files from the embedding generator. The benchmark concatenates features itself via `features/loader.py`, so these are not loaded.

### Feature Dimension Reference

Dimensions that change with the encoder (profile-derived):

| Config | Features | bge-large (default, 1024d) | bge-large + PCA-128 | MiniLM + PCA-128 |
|--------|----------|---------------------------|---------------------|-----------------|
| M3 | BERT title | 1024 | 128 | 128 |
| M4 | profile | 1024 | 128 | 128 |
| M7 | profile + mood | 1034 | 138 | 138 |
| M8 | profile + mood + themes | 1562 | 666 | 666 |

Dimensions that stay fixed regardless of encoder:

| Config | Features | Dimension |
|--------|----------|-----------|
| M2 | genome PCA | 128 |
| M5 | mood | 10 |
| M6 | themes | 528 |
| M9 | genome + mood + themes | 666 |

### Verifying an Embedding Directory

Quick sanity check before running experiments:

```bash
# Check shapes of all .npy files in an embedding directory
python3 -c "
import numpy as np, json
from pathlib import Path
d = Path('../embedding_generator/output')  # or any EMBEDDING_DIR path
for f in sorted(d.glob('*.npy')):
    a = np.load(f)
    print(f'{f.name:35s} {str(a.shape):>15s}  dtype={a.dtype}')
ids = json.load(open(d / 'movie_id_index.json'))
print(f\"{'movie_id_index.json':35s} {len(ids)} IDs\")
meta = json.load(open(d / 'embedding_metadata.json'))
print(f\"{'Encoder':35s} {meta.get('sentence_model', 'N/A')}\")
print(f\"{'PCA applied':35s} {meta.get('profile_pca_applied', 'N/A')}\")
"
```

Expected output (default bge-large-en-v1.5, no PCA):

```
combined_features.npy                   (10381, 1034)  dtype=float32
combined_full.npy                       (10381, 1562)  dtype=float32
genome_embeddings.npy                   (10381, 128)   dtype=float32
mood_vectors.npy                        (10381, 10)    dtype=float32
profile_embeddings.npy                  (10381, 1024)  dtype=float32
theme_matrix.npy                        (10381, 528)   dtype=float32
Encoder                                 BAAI/bge-large-en-v1.5
PCA applied                             False
```

## Project Structure

```
benchmark/
├── config.py                      # Paths, hyperparameters, experiment configs
├── evaluate.py                    # Full-ranking metrics (NDCG, Recall, HR, MRR)
├── train.py                       # BPR training loop with early stopping
├── run_experiment.py              # Single experiment runner
├── run_ablation.py                # Full ablation table (all configs × 5 seeds)
├── run_quick.sh                   # Quick run: BPR-MF + LightGCN-SF, 10 epochs
├── data/
│   ├── preprocess.py              # ML-20M → filtered train/val/test splits
│   ├── dataset.py                 # PyTorch Dataset + BPR negative sampling
│   └── processed/                 # Train/val/test CSVs + ID mappings
├── features/
│   ├── loader.py                  # Load .npy embeddings, align to benchmark IDs
│   └── bert_baseline.py           # Generate BERT(title+genre) baseline
├── models/
│   ├── bpr_mf.py                  # BPR Matrix Factorization
│   ├── lightgcn.py                # LightGCN + LightGCN-SF (with side features)
│   ├── simgcl.py                  # SimGCL (noise-based contrastive)
│   ├── xsimgcl.py                 # XSimGCL (cross-layer contrastive)
│   ├── lightgcl.py                # LightGCL (SVD-based contrastive)
│   ├── kar.py                     # KAR (knowledge-augmented, MoE adapter)
│   └── sasrec.py                  # SASRec (sequential, for future use)
├── external/
│   ├── RLMRec/                    # Official RLMRec repo (patched for ML-20M)
│   ├── prepare_rlmrec_data.py     # Data adapter for RLMRec format
│   └── run_rlmrec.sh              # Run script for RLMRec experiments
├── analysis/                      # Post-experiment analysis scripts
└── requirements.txt
```

## References

- Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." UAI 2009.
- He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
- Yu et al. "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation." SIGIR 2022. (SimGCL)
- Yu et al. "XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation." TKDE 2023.
- Cai et al. "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation." ICLR 2023.
- Wei et al. "RLMRec: Representation Learning with Large Language Models for Recommendation." WWW 2024.
- Xi et al. "Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models." KDD 2024. (KAR)
- Krichene & Rendle. "On Sampled Metrics for Item Recommendation." KDD 2020.
- Rendle et al. "Are We Really Making Much Progress? Revisiting, Benchmarking and Refining the Evaluation of Recommender Systems." RecSys 2019.
- Harper & Konstan. "The MovieLens Datasets: History and Context." ACM TIIS 2015.
