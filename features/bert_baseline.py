#!/usr/bin/env python3
"""
Generate BERT(title+genre) baseline embeddings.

Encodes "Title | Genre1, Genre2, ..." using the same sentence-transformer
as the profile embeddings (default: bge-large-en-v1.5, 1024-dim, no PCA).
This controls for the embedding method — isolating the value of LLM-generated
profile content vs. naive title text.
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MOVIES_CSV, MOVIE_ID_INDEX, EMBED_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(__file__).parent.parent.parent / "embedding_generator" / "output" / "bge-large-v1.5" / "bert_title_embeddings.npy"


def main():
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    # Load movie metadata
    logger.info("Loading movies.csv...")
    movies = pd.read_csv(MOVIES_CSV)
    movies["genres_clean"] = movies["genres"].str.replace("|", ", ")

    # Load genome movie ID index (same order as other embeddings)
    with open(MOVIE_ID_INDEX) as f:
        movie_ids = json.load(f)

    # Build title+genre text for each movie
    mid_to_text = {}
    for _, row in movies.iterrows():
        mid = row["movieId"]
        title = row["title"]
        genres = row["genres_clean"]
        mid_to_text[mid] = f"{title} | {genres}"

    texts = [mid_to_text.get(mid, "Unknown") for mid in movie_ids]
    logger.info(f"Prepared {len(texts)} texts for encoding")

    # Encode with same model as profile embeddings
    model_name = "BAAI/bge-large-en-v1.5"
    logger.info(f"Loading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info("Encoding titles...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Raw shape: {embeddings.shape}")

    # PCA to match profile embedding dims
    if EMBED_DIM < embeddings.shape[1]:
        logger.info(f"Applying PCA: {embeddings.shape[1]} → {EMBED_DIM}")
        pca = PCA(n_components=EMBED_DIM, random_state=42)
        embeddings = pca.fit_transform(embeddings)
        embeddings = normalize(embeddings)
        logger.info(f"PCA variance: {sum(pca.explained_variance_ratio_):.4f}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)
    logger.info(f"Saved: {OUTPUT_PATH} {embeddings.shape}")


if __name__ == "__main__":
    main()
