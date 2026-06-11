from functools import lru_cache
import numpy as np
from sentence_transformers import CrossEncoder
from src.config import CROSS_ENCODER_MODEL, TOP_K_RERANK


@lru_cache(maxsize=1)
def _get_model():
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank_chunks(query: str, chunks: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    model = _get_model()
    chunks = [dict(c) for c in chunks]  # shallow copy to avoid mutating caller's dicts
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    scores = np.atleast_1d(scores)  # handle scalar return from model.predict
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    # Stubs sort after all content chunks regardless of score — the
    # cross-encoder rates keyword-dense titles highly, but stubs only
    # signal topic presence and must never displace citable content.
    ranked = sorted(chunks, key=lambda c: (bool(c.get("is_stub")), -c["rerank_score"]))
    return ranked[:top_k]
