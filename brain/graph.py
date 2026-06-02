"""
Usage: uv run python brain/graph.py

Generates one Obsidian markdown note per indexed document, with [[wikilinks]]
to semantically related documents. Uses mutual top-K filtering: a link between
A and B is only created if both A considers B a top candidate AND B considers
A a top candidate. This prunes weak/asymmetric connections that are caused by
shared academic vocabulary rather than genuine conceptual overlap.

Run after brain/ingest.py. Re-running is safe — notes are overwritten.
"""
import json
from collections import defaultdict
from pathlib import Path
from brain.config import INGESTION_LOG, NOTES_DIR, RAW_DIR, GRAPH_TOP_K, GRAPH_MIN_SIMILARITY
from brain.index import get_collection


def _note_title(filename: str) -> str:
    """Strip extension for use as Obsidian note title and wikilink target."""
    return Path(filename).stem


def _find_candidates(collection, filename: str) -> dict[str, dict]:
    """Return top GRAPH_TOP_K candidate related documents, keyed by filename.

    Queries with every chunk embedding from this document. External documents
    are ranked by frequency × average similarity so large docs with partial
    overlap still surface their strongest connections.
    """
    result = collection.get(
        where={"filename": filename},
        include=["embeddings", "metadatas"],
    )
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return {}

    n_results = min(20, collection.count())
    query_results = collection.query(
        query_embeddings=embeddings,
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    doc_scores: dict[str, list[float]] = defaultdict(list)
    doc_course: dict[str, str] = {}

    for metadatas, distances in zip(query_results["metadatas"], query_results["distances"]):
        for meta, dist in zip(metadatas, distances):
            other_file = meta.get("filename", "")
            if other_file == filename:
                continue
            doc_scores[other_file].append(1.0 - dist)
            doc_course[other_file] = meta.get("course", "")

    ranked = []
    for other_file, scores in doc_scores.items():
        frequency = len(scores)
        avg_sim = sum(scores) / frequency
        if avg_sim < GRAPH_MIN_SIMILARITY:
            continue
        ranked.append({
            "filename": other_file,
            "course": doc_course[other_file],
            "similarity": round(avg_sim, 3),
            "frequency": frequency,
        })

    ranked.sort(key=lambda x: x["frequency"] * x["similarity"], reverse=True)
    return {r["filename"]: r for r in ranked[:GRAPH_TOP_K]}


def _apply_mutual_filter(all_candidates: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    """Keep only links that are confirmed by both sides.

    A link A → B is kept only if B also has A in its candidate list.
    Weak connections are almost always asymmetric — one doc "sees" the
    other as relevant but not vice versa. Mutual confirmation is a strong
    signal of genuine conceptual overlap.
    """
    mutual_links: dict[str, list[dict]] = {fname: [] for fname in all_candidates}

    for filename, candidates in all_candidates.items():
        for other_filename, link_data in candidates.items():
            # only keep if the other document also listed this one as a candidate
            if filename in all_candidates.get(other_filename, {}):
                mutual_links[filename].append(link_data)

    return mutual_links


def _write_note(note_path: Path, filename: str, course: str, chunk_count: int, related: list[dict]):
    title = _note_title(filename)
    lines = [
        f"# {title}",
        "",
        f"**Course:** {course}",
        f"**Chunks indexed:** {chunk_count}",
        "",
        "## Related documents",
        "",
    ]
    if related:
        for doc in sorted(related, key=lambda x: x["similarity"], reverse=True):
            link_title = _note_title(doc["filename"])
            # full vault-relative path so Obsidian resolves the link correctly
            link_path = f"notes/{doc['course']}/{link_title}"
            lines.append(
                f"- [[{link_path}|{link_title}]] "
                f"({doc['course']}, {doc['frequency']} chunk matches, avg sim {doc['similarity']})"
            )
    else:
        lines.append("_No related documents found._")

    lines.append("")
    note_path.write_text("\n".join(lines), encoding="utf-8")


def run_graph():
    if not INGESTION_LOG.exists():
        print("No ingestion log found. Run brain/ingest.py first.")
        return

    log = json.loads(INGESTION_LOG.read_text())
    if not log:
        print("Nothing indexed yet. Run brain/ingest.py first.")
        return

    collection = get_collection()
    NOTES_DIR.mkdir(exist_ok=True)

    # only generate notes for files that live inside raw/
    indexed_files = [
        Path(p) for p in log
        if Path(p).is_relative_to(RAW_DIR)
    ]

    if not indexed_files:
        print("No raw/ documents found in index. Run brain/ingest.py first.")
        return

    print(f"Computing candidates for {len(indexed_files)} documents...")

    # Phase 1: compute candidate links for every document
    doc_meta: dict[str, dict] = {}
    all_candidates: dict[str, dict[str, dict]] = {}

    for file_path in indexed_files:
        filename = file_path.name
        meta_result = collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        if not meta_result["ids"]:
            continue
        doc_meta[filename] = {
            "chunk_count": len(meta_result["ids"]),
            "course": meta_result["metadatas"][0].get("course", "Unknown"),
        }
        all_candidates[filename] = _find_candidates(collection, filename)

    # Phase 2: mutual top-K filter — only keep bidirectional links
    print("Applying mutual top-K filter...")
    mutual_links = _apply_mutual_filter(all_candidates)

    total_links = sum(len(v) for v in mutual_links.values())
    print(f"  {total_links} mutual links across {len(indexed_files)} documents")

    # Phase 3: write notes
    for filename, related in mutual_links.items():
        meta = doc_meta.get(filename, {})
        course = meta.get("course", "Unknown")
        chunk_count = meta.get("chunk_count", 0)

        course_dir = NOTES_DIR / course
        course_dir.mkdir(exist_ok=True)
        note_path = course_dir / f"{_note_title(filename)}.md"
        _write_note(note_path, filename, course, chunk_count, related)
        print(f"  {_note_title(filename)}.md -> {len(related)} mutual links")

    print(f"\nDone. {len(indexed_files)} notes written to notes/")
    print("Open Obsidian and switch to Graph View to see connections.")


if __name__ == "__main__":
    run_graph()
