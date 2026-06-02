"""
Usage: uv run python brain/graph.py

Generates one Obsidian markdown note per indexed PDF, with [[wikilinks]] to
the most semantically similar other documents. Run after brain/ingest.py.
Re-running is safe — notes are overwritten with fresh similarity data.
"""
import json
import numpy as np
from pathlib import Path
from brain.config import INGESTION_LOG, NOTES_DIR, GRAPH_TOP_K
from brain.index import get_collection


def _note_title(filename: str) -> str:
    """Strip .pdf extension for use as Obsidian note title and wikilink target."""
    return Path(filename).stem


def _document_embedding(collection, filename: str) -> np.ndarray | None:
    """Average all chunk embeddings for a file to get a document-level vector."""
    result = collection.get(
        where={"filename": filename},
        include=["embeddings"],
    )
    if not result["embeddings"]:
        return None
    return np.mean(result["embeddings"], axis=0).tolist()


def _find_related(collection, doc_embedding: list[float], exclude_filename: str) -> list[dict]:
    """Return top GRAPH_TOP_K documents most similar to doc_embedding, excluding self."""
    results = collection.query(
        query_embeddings=[doc_embedding],
        n_results=min(50, collection.count()),
        include=["metadatas", "distances"],
    )
    seen_files = {exclude_filename}
    related = []
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        fname = metadata.get("filename", "")
        if fname in seen_files:
            continue
        seen_files.add(fname)
        related.append({
            "filename": fname,
            "course": metadata.get("course", ""),
            "similarity": round(1.0 - distance, 3),
        })
        if len(related) >= GRAPH_TOP_K:
            break
    return related


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
        for doc in related:
            link_title = _note_title(doc["filename"])
            lines.append(f"- [[{link_title}]] ({doc['course']}, similarity {doc['similarity']})")
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

        # get chunk count and course from ChromaDB metadata
        meta_result = collection.get(
            where={"filename": filename},
            include=["metadatas"],
        )
        if not meta_result["ids"]:
            print(f"  Skipping {filename} — not found in ChromaDB.")
            continue

        chunk_count = len(meta_result["ids"])
        course = meta_result["metadatas"][0].get("course", "Unknown")

        doc_embedding = _document_embedding(collection, filename)
        if doc_embedding is None:
            print(f"  Skipping {filename} — no embeddings found.")
            continue

        related = _find_related(collection, doc_embedding, exclude_filename=filename)

        note_path = NOTES_DIR / f"{title}.md"
        _write_note(note_path, filename, course, chunk_count, related)
        print(f"  {title}.md → {len(related)} links")

    print(f"\nDone. {len(indexed_pdfs)} notes written to notes/")
    print("Open Obsidian and switch to Graph View to see connections.")


if __name__ == "__main__":
    run_graph()
