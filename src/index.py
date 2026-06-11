import pickle
import re
from functools import lru_cache
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from src.config import CHROMA_DIR, CHROMA_COLLECTION

# Common English stopwords — kept small and conservative so domain terms
# ("system", "process", "model") are never accidentally filtered.
_STOPWORDS = frozenset("""
a an and are as at be but by for from has have if in into is it its of on or
that the their this to was were what when where which who will with
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25: lowercase, split on non-alphanumerics (so
    punctuation never blocks a match and hyphenated terms split), and drop
    stopwords. Must be applied identically at index and query time."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=1)
def _get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    return _get_client().get_or_create_collection(
        CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: list[dict], embeddings: list[list[float]]):
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{k: v for k, v in c.items() if k not in ("text", "id")} for c in chunks],
    )


def query_dense(collection, query_vector: list[float], top_k: int,
                where: dict | None = None) -> list[dict]:
    results = collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)
    output = []
    for i, chunk_id in enumerate(results["ids"][0]):
        output.append({
            "id": chunk_id,
            "text": results["documents"][0][i],
            "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
            **results["metadatas"][0][i],
        })
    return output


def build_bm25(chunks: list[dict], bm25_path: Path):
    tokenized = [tokenize(c["text"]) for c in chunks]
    index = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump((index, tokenized, chunks), f)
    return index, tokenized


def rebuild_bm25_from_collection(collection, bm25_path: Path) -> int:
    """Rebuild the BM25 index from ChromaDB contents. Returns chunk count.

    Chroma is the single source of truth; the BM25 pickle is a derived cache
    regenerated on every ingest run, so the two stores can never drift (e.g.
    a crash mid-ingest, or purges that previously had to update both by hand).
    """
    result = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"id": cid, "text": text, **meta}
        for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    if not chunks:
        if bm25_path.exists():
            bm25_path.unlink()
        return 0
    build_bm25(chunks, bm25_path)
    return len(chunks)


def load_bm25(bm25_path: Path):
    # bm25.pkl is written by this codebase only — trust boundary is local filesystem
    with open(bm25_path, "rb") as f:
        return pickle.load(f)  # (index, tokenized_corpus, chunks)


def query_bm25(index, corpus, chunks: list[dict], query: str, top_k: int,
               course: str | None = None) -> list[dict]:
    tokens = tokenize(query)
    scores = index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for i, s in ranked:
        if s <= 0:
            break  # sorted descending — nothing scoreworthy remains
        if course and chunks[i].get("course") != course:
            continue
        results.append(
            {"id": chunks[i]["id"], "text": chunks[i]["text"], "score": float(s),
             **{k: v for k, v in chunks[i].items() if k not in ("id", "text")}}
        )
        if len(results) == top_k:
            break
    return results
