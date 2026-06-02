"""
Usage: uv run python brain/graph.py

Generates one Obsidian markdown note per indexed PDF, with [[wikilinks]] to
the most semantically similar other documents. Run after brain/ingest.py.
Re-running is safe — notes are overwritten with fresh similarity data.
"""
import json
from collections import defaultdict
from pathlib import Path
from brain.config import INGESTION_LOG, NOTES_DIR, GRAPH_TOP_K, GRAPH_MIN_SIMILARITY
from brain.index import get_collection


def _note_title(filename: str) -> str:
    """Strip .pdf extension for use as Obsidian note title and wikilink target."""
    return Path(filename).stem


def _find_related_by_chunks(collection, filename: str) -> list[dict]:
    """Find related documents by querying with every chunk of this document.

    Rather than averaging all chunk embeddings into one blurry document vector,
    each chunk gets its own similarity query. External documents are ranked by
    frequency (how many chunks matched) × average similarity — so a large slide
    deck that partially overlaps with an external reading will surface that link
    even if the rest of the deck is unrelated.
    """
    result = collection.get(
        where={"filename": filename},
        include=["embeddings", "metadatas"],
    )
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return []

    # batch query: one call with all chunk embeddings
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
        ranked.append({
            "filename": other_file,
            "course": doc_course[other_file],
            "similarity": round(avg_sim, 3),
            "frequency": frequency,
        })

    ranked.sort(key=lambda x: x["frequency"] * x["similarity"], reverse=True)
    # apply minimum similarity threshold — don't force connections that aren't real
    ranked = [r for r in ranked if r["similarity"] >= GRAPH_MIN_SIMILARITY]
    return ranked[:GRAPH_TOP_K]


def _course_tag(course: str) -> str:
    """Convert 'IS 6410' → 'IS-6410' for use as an Obsidian tag."""
    return course.replace(" ", "-")


def _write_note(note_path: Path, filename: str, course: str, chunk_count: int, related: list[dict]):
    title = _note_title(filename)
    tag = _course_tag(course)
    lines = [
        "---",
        f"course: {course}",
        f"tags: [{tag}]",
        "---",
        "",
        f"# {title}",
        "",
        f"**Course:** {course}",
        f"**Chunks indexed:** {chunk_count}",
        "",
        "## Related documents",
        "",
    ]
    if related:
        for doc in related:
            link_title = _note_title(doc["filename"])
            link_path = f"{doc['course']}/{link_title}"
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

    indexed_pdfs = [Path(p) for p in log]
    print(f"Generating notes for {len(indexed_pdfs)} documents...")

    for pdf_path in indexed_pdfs:
        filename = pdf_path.name
        title = _note_title(filename)

        meta_result = collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        if not meta_result["ids"]:
            print(f"  Skipping {filename} — not found in ChromaDB.")
            continue

        chunk_count = len(meta_result["ids"])
        course = meta_result["metadatas"][0].get("course", "Unknown")

        related = _find_related_by_chunks(collection, filename)

        course_dir = NOTES_DIR / course
        course_dir.mkdir(exist_ok=True)
        note_path = course_dir / f"{title}.md"
        _write_note(note_path, filename, course, chunk_count, related)
        print(f"  {title}.md -> {len(related)} links")

    print(f"\nDone. {len(indexed_pdfs)} notes written to notes/")
    print("Open Obsidian and switch to Graph View to see connections.")


if __name__ == "__main__":
    run_graph()
