"""
Benchmark configuration.
All paths, hyperparameters, and experiment configs in one place.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CODE_ROOT = PROJECT_ROOT.parent

# ML-20M raw data
PROFILE_GEN_ROOT = CODE_ROOT / "profile_generator" / "llm-movie-profiler-v1-20260402"
ML20M_DIR = PROFILE_GEN_ROOT / "data" / "ml-20m"
RATINGS_CSV = ML20M_DIR / "ratings.csv"
MOVIES_CSV = ML20M_DIR / "movies.csv"
GENOME_SCORES_CSV = ML20M_DIR / "genome-scores.csv"
GENOME_TAGS_CSV = ML20M_DIR / "genome-tags.csv"

# Pre-computed embeddings (overridable via EMBEDDING_DIR env var for remote packages)
EMBEDDING_DIR = Path(os.environ.get("EMBEDDING_DIR", str(CODE_ROOT / "embedding_generator" / "output" / "bge-large-v1.5")))

PROFILE_EMB_NPY = EMBEDDING_DIR / "profile_embeddings.npy"
MOOD_VECTORS_NPY = EMBEDDING_DIR / "mood_vectors.npy"
THEME_MATRIX_NPY = EMBEDDING_DIR / "theme_matrix.npy"
GENOME_EMB_NPY = EMBEDDING_DIR / "genome_embeddings.npy"
GENOME_RAW_EMB_NPY = EMBEDDING_DIR / "genome_raw_embeddings.npy"
COMBINED_NPY = EMBEDDING_DIR / "combined_features.npy"
COMBINED_FULL_NPY = EMBEDDING_DIR / "combined_full.npy"
MOVIE_ID_INDEX = EMBEDDING_DIR / "movie_id_index.json"
EMBEDDING_METADATA = EMBEDDING_DIR / "embedding_metadata.json"

# Benchmark outputs
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ──────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
POSITIVE_THRESHOLD = 3.5       # rating >= this → positive interaction
K_CORE = 10                    # iterative k-core filtering threshold

# Temporal split timestamps (Unix epoch)
# Train: before 2014-01-01, Val: 2014-01-01 to 2014-07-01, Test: after 2014-07-01
TRAIN_END_TS = 1388534400      # 2014-01-01 00:00:00 UTC
VAL_END_TS = 1404172800        # 2014-07-01 00:00:00 UTC

# ──────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS (tuned on validation set)
# ──────────────────────────────────────────────────────────────────────────────
EMBED_DIM = 128                # shared embedding dimension for all models
LIGHTGCN_LAYERS = 3            # number of graph convolution layers
LIGHTGCN_DROPOUT = 0.0         # no dropout (per original LightGCN paper)
LIGHTGCL_SVD_Q = 5             # SVD rank for LightGCL contrastive view
KAR_N_EXPERTS = 4              # number of expert MLPs in KAR hybrid adapter

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5            # L2 regularization
BATCH_SIZE = 32768
NUM_EPOCHS = 100
PATIENCE = 20                  # early stopping patience (epochs without val improvement)
NUM_NEGATIVES = 1              # negatives per positive for BPR

# Evaluation
EVAL_BATCH_SIZE = 256          # users per evaluation batch
TOP_K = [10, 20, 50]         # K values for NDCG@K, Recall@K, HR@K
# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGS
# ──────────────────────────────────────────────────────────────────────────────
NUM_SEEDS = 5
SEEDS = [12, 42, 123, 456, 789, 2026, 12]

# Hyperparameter search grid (for tuning)
HP_GRID = {
    "lr": [1e-4, 5e-4, 1e-3, 5e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "n_layers": [2, 3, 4],
    "batch_size": [1024, 2048, 4096],
}

# Side feature configurations for ablation (feature_name → npy file or list of npy files)
FEATURE_CONFIGS = {
    "none":           [],                                              # M1: ID only
    "genome":         ["genome"],                                      # M2: genome PCA (128-dim)
    "genome_raw":     ["genome_raw"],                                  # M2b: genome no PCA (1128-dim)
    "bert_title":     ["bert_title"],                                  # M3: BERT(title+genre)
    "llm_profile":    ["profile"],                                     # M4: LLM profile
    "llm_mood":       ["mood"],                                        # M5: mood
    "llm_themes":     ["themes"],                                      # M6: themes
    "llm_prof_mood":  ["profile", "mood"],                             # M7: profile + mood
    "llm_all":        ["profile", "mood", "themes"],                   # M8: all LLM
    "genome_llm":     ["genome", "mood", "themes"],                    # M9: genome + structured LLM
}

# Cold-start buckets (by number of training interactions)
COLD_START_BUCKETS = {
    "cold":   (0, 10),      # <10 training interactions
    "medium": (10, 50),     # 10-50
    "warm":   (50, float("inf")),  # >50
}
