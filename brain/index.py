import pickle
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from brain.config import CHROMA_DIR, CHROMA_COLLECTION


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(CHROMA_COLLECTION)


def upsert_chunks(collection, chunks: list[dict], embeddings: list[list[float]]):
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{k: v for k, v in c.items() if k not in ("text", "id")} for c in chunks],
    )


def query_dense(collection, query_vector: list[float], top_k: int) -> list[dict]:
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
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
    tokenized = [c["text"].lower().split() for c in chunks]
    index = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump((index, tokenized, chunks), f)
    return index, tokenized


def load_bm25(bm25_path: Path):
    with open(bm25_path, "rb") as f:
        return pickle.load(f)  # (index, tokenized_corpus, chunks)


def query_bm25(index, corpus, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    tokens = query.lower().split()
    scores = index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"id": chunks[i]["id"], "text": chunks[i]["text"], "score": float(s),
         **{k: v for k, v in chunks[i].items() if k not in ("id", "text")}}
        for i, s in ranked if s > 0
    ]
