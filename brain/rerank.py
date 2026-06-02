from functools import lru_cache
from sentence_transformers import CrossEncoder
from brain.config import CROSS_ENCODER_MODEL, TOP_K_RERANK


@lru_cache(maxsize=1)
def _get_model():
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank_chunks(query: str, chunks: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
