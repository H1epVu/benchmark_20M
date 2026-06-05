# Direction 4 — Sequential × Content × Density

> **Research question:** Does injecting LLM-generated content features into sequential recommendation models reduce their density sensitivity — and if so, at which architectural injection point?

---

## 1. Problem

### What is the problem?

Collaborative filtering (CF) models augmented with LLM content features (M7 = profile + mood) show a **6.8× performance decay** as dataset density decreases — strong gains at dense datasets (ML-20M), near-zero gains at sparse datasets (Amazon-Books). Sequential models like SASRec are theoretically better suited for sparse settings because they exploit item order rather than relying on co-occurrence density. However, it is unknown whether content features can be effectively fused into sequential architectures, and whether such fusion makes sequential models more robust to density variation.

### Why does the problem exist?

Sequential models encode user preference as a **causal sequence of interactions** rather than a user embedding trained on all historical co-occurrences. This means:

- Standard additive injection (item\_emb + MLP(features)) used in LightGCN-SF cannot be directly applied — SASRec processes sequences token-by-token, so injection must respect the temporal structure.
- The transformer architecture has multiple semantically distinct stages (input embedding → self-attention → FFN → output scoring), each offering a different integration point with different inductive biases.
- No prior work has systematically ablated these injection points on a density-stratified benchmark.

### Different branches of the problem

| Branch | Question |
|---|---|
| **Which injection point?** | Input, FFN, or output scoring — each has different expressivity and gradient flow |
| **Does it help across densities?** | Does injection flatten the 6.8× decay, or is sequential intrinsically density-limited? |
| **Is SASRec representative?** | Do stronger models (BERT4Rec, DuoRec) confirm the same pattern at the dense endpoint? |

---

## 2. Solution

### What is the solution?

Three **independent content injection points** are implemented in SASRec, each controlled by a single `injection_mode` flag:

```
injection_mode: "none" | "input" | "ffn" | "output" | combinations | "all"
```

Each active injection point has its own MLP projection (`feature_dim → embed_dim`) with separate weights, so ablation is clean — activating one point does not share parameters with another.

### Why solved this way?

- **Independent MLPs per point**: ensures that comparing `--injection input` vs `--injection ffn` is a true ablation — the same number of parameters, different placement. Shared weights would conflate injection position with capacity.
- **Additive injection** (emb + MLP(features)): consistent with LightGCN-SF's proven integration strategy. Keeps the embedding dimension fixed at `d=128` across all experiments.
- **Right-padding sequences**: SASRec uses a causal attention mask. Left-padding causes the first (padding) position to attend only to itself → `softmax(-∞) = NaN`. Right-padding avoids this by placing real items at the start.
- **Item ID shift +1**: SASRec reserves token ID 0 as the padding token. The benchmark uses 0-indexed items (0..n\_items−1), so all item IDs are shifted +1 at both training and evaluation time.
- **`train_sequences` vs `train_val_sequences`**: val evaluation uses only training history as context; test evaluation uses train+val history. Using test-time history during val evaluation would leak future signal.

### Design decisions

| Decision | Rationale |
|---|---|
| `embed_dim = 128` | Uniform across all models in the benchmark — enables direct comparison |
| `n_blocks = 2, n_heads = 2` | Standard modern SASRec; paper original used `d=50, n_heads=1` but benchmark uses unified capacity |
| `max_seq_len = 50` | SASRec paper default; ML-20M users average ~90 interactions — increasing to 200 captures more context at the cost of ~2× memory |
| BPR loss | Consistent with all other models in benchmark; original SASRec used BCE |
| Full ranking evaluation | All 9,906 items scored per user — avoids sampled evaluation bias (Krichene & Rendle, KDD 2020) |
| 5 seeds | Required for paired t-test statistical comparison |
| AMP (mixed precision) | Halves VRAM, ~1.5× faster; numerically equivalent for this task |

### Edge cases

- **475 items filtered by 10-core** (10,381 genome movies → 9,906 benchmark items): these receive a zero feature vector. The model learns to treat zero-feature items as ID-only, which is the correct fallback.
- **Users with no training sequence** (new users at val/test time): `predict()` returns scores based on zero-length context — effectively random ranking. These users exist in val/test due to temporal split but are rare.
- **`injection_mode = "all"`**: activates all three projections simultaneously (3× parameter overhead). Included for completeness but not part of the core Direction 4 ablation spec.

---

## 3. Broader Context

### Why does this matter?

The parent paper (CIKM/NeurIPS submission) established a **density law**: regularizer paradigms gain monotonically with sparsity; injection paradigms (M7 on LightGCN-SF) decay 6.8× from dense to sparse. The law is *descriptive* — it characterizes what happens but does not explain *why* or offer a fix.

Direction 4 tests whether the sequential axis changes this picture:

- If content injection into SASRec **flattens the decay** → sequential architecture is the key, and the density law can be partially circumvented by architectural choice.
- If content injection **does not help** at sparse densities → sequential models are intrinsically density-limited regardless of content, and the law holds across architectures. This is a publishable negative result that strengthens the law's generality.

Either outcome advances the research: one converts the law from descriptive to actionable; the other extends its domain of validity.

### What will the changes affect?

| Component | Impact |
|---|---|
| `models/sasrec.py` | New — SASRec backbone with 3 injection points |
| `data/dataset.py` | Added `SequenceDataset`, `train/train_val_sequences` |
| `train.py` | Added `train_epoch_seq`, `train_seq_model` with AMP + tqdm |
| `run_experiment.py` | Added `--model sasrec`, `--injection`, `--dataset`, `--seq-batch-size` |
| `config.py` | Added `MAX_SEQ_LEN`, `SEQ_BATCH_SIZE`, `DATASET_CONFIGS` |
| Results | Saved to `results/sasrec__<dataset>__<injection>__<features>__seed<N>/results.json` |

---

## Experiment Plan

### Phase 1 — SASRec injection ablation (ML-20M, available now)

```bash
# 4 configs × 5 seeds = 20 runs
python run_experiment.py --model sasrec --dataset ml20m --features none         --injection none   --seed {42,123,456,789,2026}
python run_experiment.py --model sasrec --dataset ml20m --features llm_prof_mood --injection input  --seed {42,123,456,789,2026}
python run_experiment.py --model sasrec --dataset ml20m --features llm_prof_mood --injection ffn    --seed {42,123,456,789,2026}
python run_experiment.py --model sasrec --dataset ml20m --features llm_prof_mood --injection output --seed {42,123,456,789,2026}
```

### Phase 2 — Density ablation (3 more datasets, requires data prep)

```bash
# 4 configs × 3 datasets × 5 seeds = 60 runs
# datasets: sub_ml20m | ml1m | amazon
python run_experiment.py --model sasrec --dataset <dataset> --features {none,llm_prof_mood} --injection {none,input,ffn,output} --seed {42,123,456,789,2026}
```

### Phase 3 — Stronger baselines at dense endpoint (requires BERT4Rec + DuoRec implementation)

```bash
# 2 models × 2 configs × 5 seeds = 20 runs
python run_experiment.py --model bert4rec --dataset ml20m --features {none,llm_prof_mood} --injection <best> --seed {42,123,456,789,2026}
python run_experiment.py --model duorec   --dataset ml20m --features {none,llm_prof_mood} --injection <best> --seed {42,123,456,789,2026}
```

**Total: 100 runs across all phases.**

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | required | `sasrec` (+ `bert4rec`, `duorec` in Phase 3) |
| `--dataset` | `ml20m` | `ml20m` \| `sub_ml20m` \| `ml1m` \| `amazon` |
| `--features` | `none` | `none` (baseline) \| `llm_prof_mood` (M7) |
| `--injection` | `none` | `none` \| `input` \| `ffn` \| `output` |
| `--seq-batch-size` | `512` | L4=4096, T4=1024, A100=8192 |
| `--epochs` | `100` | Max epochs (early stopping at patience=20) |
| `--no-amp` | off | Disable mixed precision (enabled by default on CUDA) |

For full benchmark specification (all models, preprocessing, evaluation protocol), see [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).
