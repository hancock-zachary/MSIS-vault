from collections import defaultdict
from brain.config import RRF_K, TOP_K_RETRIEVAL, BM25_PATH
from brain.embed import embed_text
from brain.index import get_collection, query_dense, load_bm25, query_bm25


def reciprocal_rank_fusion(ranked_lists: list[list[dict]]) -> list[dict]:
    """
    Fuse multiple ranked lists using reciprocal rank fusion (RRF).

    For each chunk appearing in the lists, score += 1 / (RRF_K + rank)
    where rank is 1-indexed position in that list.

    Args:
        ranked_lists: List of ranked chunk lists. Each chunk is a dict with at minimum an "id" key.

    Returns:
        Merged chunks sorted by RRF score (highest first), with "rrf_score" added to each.
    """
    scores = defaultdict(float)
    chunk_store = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk["id"]] += 1.0 / (RRF_K + rank)
            chunk_store[chunk["id"]] = chunk
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for cid in sorted_ids:
        c = dict(chunk_store[cid])
        c["rrf_score"] = scores[cid]
        result.append(c)
    return result


def deduplicate_by_id(chunks: list[dict]) -> list[dict]:
    """
    Remove duplicates by keeping only the first occurrence of each id.

    Args:
        chunks: List of chunks with an "id" field.

    Returns:
        List with duplicates removed, preserving order of first occurrence.
    """
    seen = set()
    result = []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            result.append(c)
    return result


def hybrid_retrieve(query_variants: list[str], top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    Perform hybrid retrieval: dense + sparse for each query variant, fused with RRF.

    For each query variant:
    - Run dense (embedding) retrieval
    - Run sparse (BM25) retrieval
    Both results feed into reciprocal rank fusion.

    Args:
        query_variants: List of query rewrite/reformulations.
        top_k: Top K results per retrieval method per variant.

    Returns:
        Deduplicated and RRF-fused chunk list.
    """
    collection = get_collection()
    bm25_index, bm25_corpus, bm25_chunks = load_bm25(BM25_PATH)

    all_ranked_lists = []
    for variant in query_variants:
        vec = embed_text(variant)
        dense_results = query_dense(collection, vec, top_k)
        sparse_results = query_bm25(bm25_index, bm25_corpus, bm25_chunks, variant, top_k)
        all_ranked_lists.extend([dense_results, sparse_results])

    fused = reciprocal_rank_fusion(all_ranked_lists)
    return deduplicate_by_id(fused)
