"""
vector_store.py — ChromaDB operations: add chunks, query top-K
"""
import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any


def get_client() -> chromadb.Client:
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_store")
    return chromadb.PersistentClient(path=db_path)


def get_collection(name: str = "rag_docs") -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    source: str,
    doc_id: str
) -> None:
    """Add text chunks with their embeddings to ChromaDB."""
    collection = get_collection()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


def query_chunks(
    query_embedding: List[float],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """Retrieve top-K most relevant chunks."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()) if collection.count() > 0 else 1,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(1 - dist, 4)
            })
    return hits


def delete_document(doc_id: str) -> None:
    """Remove all chunks of a document from the store."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def count_chunks() -> int:
    return get_collection().count()


def reset_store() -> None:
    """Delete all documents from the collection."""
    client = get_client()
    client.delete_collection("rag_docs")
