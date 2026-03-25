"""
embedder.py — Generate embeddings using sentence-transformers
"""
import os
from sentence_transformers import SentenceTransformer
from typing import List

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return list of embedding vectors for given texts."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
