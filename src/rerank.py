from functools import lru_cache
import numpy as np
from sentence_transformers import CrossEncoder
from src.config import CROSS_ENCODER_MODEL, SOURCE_TYPE_RERANK_BONUS, TOP_K_RERANK


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
        # Source-type trust bonus: professor material outranks student work
        # when relevance is comparable. Additive — logits can be negative.
        bonus = SOURCE_TYPE_RERANK_BONUS.get(chunk.get("source_type", "unknown"), 0.0)
        chunk["rerank_score"] = float(score) + bonus
    # Stubs sort after all content chunks regardless of score — the
    # cross-encoder rates keyword-dense titles highly, but stubs only
    # signal topic presence and must never displace citable content.
    ranked = sorted(chunks, key=lambda c: (bool(c.get("is_stub")), -c["rerank_score"]))
    return ranked[:top_k]
