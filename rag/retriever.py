"""
retriever.py — High-level retrieval logic
"""
import os
from typing import List, Dict, Any
from rag.embedder import embed_query
from rag.vector_store import query_chunks


def retrieve(query: str) -> List[Dict[str, Any]]:
    """Embed query and retrieve top-K relevant chunks."""
    top_k = int(os.getenv("TOP_K_CHUNKS", 4))
    q_embedding = embed_query(query)
    return query_chunks(q_embedding, top_k=top_k)


def format_context(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[Source {i}: {hit['source']}]\n{hit['text']}")
    return "\n\n---\n\n".join(parts)
