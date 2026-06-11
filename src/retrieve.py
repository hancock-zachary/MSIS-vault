from collections import defaultdict
from src.config import (
    NEIGHBOR_PAGE_WINDOW, RRF_K, STUB_RRF_MULTIPLIER, TOP_K_RETRIEVAL, BM25_PATH,
)
from src.embed import embed_text
from src.index import get_collection, query_dense, load_bm25, query_bm25


def _chunk_range(chunk: dict) -> tuple[int, int]:
    start = chunk.get("page_start", chunk.get("page", 0))
    end = chunk.get("page_end", chunk.get("page", 0))
    return start, end


def _reading_order(chunk: dict) -> tuple[int, int]:
    return _chunk_range(chunk)[0], chunk.get("chunk_index", 0)


def select_neighbors(chunk: dict, same_file_chunks: list[dict],
                     window: int = NEIGHBOR_PAGE_WINDOW) -> list[dict]:
    """Non-stub chunks whose page range touches the chunk's range ± window,
    in reading order. Same-page siblings (semantic sub-chunks) count too."""
    start, end = _chunk_range(chunk)
    lo, hi = start - window, end + window
    neighbors = []
    for candidate in same_file_chunks:
        if candidate["id"] == chunk["id"] or candidate.get("is_stub"):
            continue
        c_start, c_end = _chunk_range(candidate)
        if c_start <= hi and c_end >= lo:
            neighbors.append(candidate)
    neighbors.sort(key=_reading_order)
    return neighbors


def expand_neighbors(chunks: list[dict]) -> list[dict]:
    """Merge page-adjacent chunk text into each reranked winner (small-to-big).

    Retrieval and reranking stay chunk-precise; the context block grows to
    section level. Each neighbor is claimed by the first winner that reaches
    it — and winners never absorb each other — so no text appears twice.
    """
    collection = get_collection()
    by_file: dict[str, list[dict]] = {}
    for filename in {c["filename"] for c in chunks}:
        result = collection.get(where={"filename": filename}, include=["documents", "metadatas"])
        by_file[filename] = [
            {"id": cid, "text": text, **meta}
            for cid, text, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    used = {c["id"] for c in chunks}
    expanded = []
    for chunk in chunks:
        neighbors = [
            n for n in select_neighbors(chunk, by_file.get(chunk["filename"], []))
            if n["id"] not in used
        ]
        merged = dict(chunk)
        if neighbors:
            used.update(n["id"] for n in neighbors)
            parts = sorted(neighbors + [chunk], key=_reading_order)
            merged["text"] = "\n\n".join(p["text"] for p in parts)
            merged["page_start"] = min(_chunk_range(p)[0] for p in parts)
            merged["page_end"] = max(_chunk_range(p)[1] for p in parts)
        expanded.append(merged)
    return expanded


def apply_stub_penalty(chunks: list[dict]) -> list[dict]:
    """Scale down RRF scores of stub chunks and re-sort.

    Stubs (salvaged titles from garbled pages) are short and keyword-dense,
    so BM25 in particular can rank them highly. They signal topic presence
    but must not crowd real content out of the candidate pool.
    """
    result = []
    for c in chunks:
        c = dict(c)
        if c.get("is_stub"):
            c["rrf_score"] = c["rrf_score"] * STUB_RRF_MULTIPLIER
        result.append(c)
    result.sort(key=lambda c: c["rrf_score"], reverse=True)
    return result


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


def hybrid_retrieve(query_variants: list[str], top_k: int = TOP_K_RETRIEVAL,
                    course: str | None = None) -> list[dict]:
    """
    Perform hybrid retrieval: dense + sparse for each query variant, fused with RRF.

    For each query variant:
    - Run dense (embedding) retrieval
    - Run sparse (BM25) retrieval if available
    Both results feed into reciprocal rank fusion.
    Falls back to dense-only if BM25 index is not yet built.

    Args:
        query_variants: List of query rewrite/reformulations.
        top_k: Top K results per retrieval method per variant.
        course: When set, restrict both retrieval methods to this course.

    Returns:
        Deduplicated and RRF-fused chunk list.
    """
    collection = get_collection()
    where = {"course": course} if course else None

    try:
        bm25_index, bm25_corpus, bm25_chunks = load_bm25(BM25_PATH)
        bm25_available = True
    except (FileNotFoundError, Exception):
        bm25_available = False

    all_ranked_lists = []
    for variant in query_variants:
        vec = embed_text(variant)
        dense_results = query_dense(collection, vec, top_k, where=where)
        all_ranked_lists.append(dense_results)
        if bm25_available:
            sparse_results = query_bm25(
                bm25_index, bm25_corpus, bm25_chunks, variant, top_k, course=course
            )
            all_ranked_lists.append(sparse_results)

    fused = reciprocal_rank_fusion(all_ranked_lists)
    return apply_stub_penalty(deduplicate_by_id(fused))
