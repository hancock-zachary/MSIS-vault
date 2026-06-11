"""
Usage: uv run python src/graph.py

Builds and maintains a persistent wiki in wiki/ — one page per indexed document,
plus wiki/index.md (content catalog) and wiki/log.md (append-only run log).

Each wiki page contains:
  - YAML frontmatter (course, chunks, updated — queryable via Obsidian Dataview)
  - Key excerpts pulled from the top chunks (readable content, not just metadata)
  - Related documents as [[wikilinks]] (mutual top-K filtered)

The wiki is the persistent, compounding artifact described in Karpathy's LLM Wiki
pattern: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

Run after src/ingest.py. Re-running is safe — pages are overwritten, stale pages
are deleted, and index.md / log.md are kept current.
"""
import json
import textwrap
from collections import defaultdict
from datetime import date
from pathlib import Path

from src.config import INGESTION_LOG, WIKI_DIR, RAW_DIR, GRAPH_TOP_K, GRAPH_MIN_SIMILARITY
from src.index import get_collection

# Maximum characters to show per excerpt block on a wiki page.
_EXCERPT_MAX_CHARS = 600
# Number of chunk excerpts to include on each page.
_EXCERPTS_PER_PAGE = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _note_title(filename: str) -> str:
    """Strip extension for use as Obsidian note title and wikilink target."""
    return Path(filename).stem


def _first_sentence(text: str, max_chars: int = 120) -> str:
    """Return roughly the first sentence of text, capped at max_chars."""
    text = " ".join(text.split())  # collapse whitespace
    for sep in (".", "!", "?", "\n"):
        idx = text.find(sep)
        if 0 < idx < max_chars:
            return text[: idx + 1].strip()
    return text[:max_chars].rstrip() + ("…" if len(text) > max_chars else "")


# ---------------------------------------------------------------------------
# Phase 1: candidate link discovery
# ---------------------------------------------------------------------------

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
    """Keep only links confirmed by both sides (bidirectional mutual top-K)."""
    mutual_links: dict[str, list[dict]] = {fname: [] for fname in all_candidates}
    for filename, candidates in all_candidates.items():
        for other_filename, link_data in candidates.items():
            if filename in all_candidates.get(other_filename, {}):
                mutual_links[filename].append(link_data)
    return mutual_links


# ---------------------------------------------------------------------------
# Phase 2: fetch excerpts from ChromaDB
# ---------------------------------------------------------------------------

def _get_excerpts(collection, filename: str, n: int = _EXCERPTS_PER_PAGE) -> list[str]:
    """Pull the first n chunk texts for this document, sorted by page number."""
    result = collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"],
    )
    if not result["ids"]:
        return []

    paired = sorted(
        zip(result["metadatas"], result["documents"]),
        key=lambda x: x[0].get("page", 0),
    )
    excerpts = []
    for _, text in paired[:n]:
        text = " ".join(text.split())  # collapse whitespace
        if len(text) > _EXCERPT_MAX_CHARS:
            text = text[:_EXCERPT_MAX_CHARS].rstrip() + "…"
        excerpts.append(text)
    return excerpts


# ---------------------------------------------------------------------------
# Phase 3: write individual wiki pages
# ---------------------------------------------------------------------------

def _write_page(
    page_path: Path,
    filename: str,
    course: str,
    chunk_count: int,
    related: list[dict],
    excerpts: list[str],
    source_type: str = "unknown",
) -> None:
    today = date.today().isoformat()
    title = _note_title(filename)

    # YAML frontmatter — queryable via Obsidian Dataview plugin
    frontmatter = textwrap.dedent(f"""\
        ---
        course: {course}
        source_type: {source_type}
        chunks: {chunk_count}
        updated: {today}
        tags: [{course.replace(" ", "-")}, indexed]
        ---
    """)

    lines = [
        frontmatter,
        f"# {title}",
        "",
    ]

    # Key excerpts — actual readable content from the source
    if excerpts:
        lines += ["## Key Excerpts", ""]
        for i, excerpt in enumerate(excerpts, 1):
            lines += [f"> **[{i}]** {excerpt}", ""]

    # Related documents — mutual top-K wikilinks
    lines += ["## Related Documents", ""]
    if related:
        for doc in sorted(related, key=lambda x: x["similarity"], reverse=True):
            link_title = _note_title(doc["filename"])
            lines.append(
                f"- [[{link_title}]] "
                f"({doc['course']} · {doc['frequency']} chunk matches · sim {doc['similarity']})"
            )
    else:
        lines.append("_No related documents found above similarity threshold._")

    lines.append("")
    page_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 4: wiki/index.md — content catalog
# ---------------------------------------------------------------------------

def _write_index(wiki_dir: Path, docs_by_course: dict[str, list[dict]]) -> None:
    """Write wiki/index.md — a catalog of every page, organized by course.

    Each entry includes a plain-text title and a one-line preview from the first chunk.
    The LLM reads this file first when answering queries so it knows what exists.
    """
    today = date.today().isoformat()
    total = sum(len(v) for v in docs_by_course.values())

    lines = [
        "# Wiki Index",
        "",
        f"_Last updated: {today} · {total} documents_",
        "",
        "> This file is auto-generated by `src/graph.py`. Do not edit manually.",
        "",
    ]

    for course in sorted(docs_by_course):
        lines += [f"## {course}", ""]
        for entry in sorted(docs_by_course[course], key=lambda x: x["title"]):
            preview = entry.get("preview", "")
            lines.append(f"- {entry['title']} — {entry['chunks']} chunks · {preview}")
        lines.append("")

    (wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase 5: wiki/log.md — append-only run log
# ---------------------------------------------------------------------------

def _append_log(
    wiki_dir: Path,
    doc_count: int,
    link_count: int,
    stale_deleted: list[str],
    orphans: list[str],
) -> None:
    """Append one entry to wiki/log.md for this graph run.

    Each entry starts with `## [DATE]` so it's grep-parseable:
        grep "^## \\[" wiki/log.md | tail -5
    """
    today = date.today().isoformat()
    log_path = wiki_dir / "log.md"

    entry_lines = [
        f"## [{today}] graph | {doc_count} documents, {link_count} mutual links",
        "",
        f"- Wrote {doc_count} wiki pages",
    ]
    if stale_deleted:
        entry_lines.append(f"- Deleted {len(stale_deleted)} stale page(s): {', '.join(stale_deleted)}")
    if orphans:
        entry_lines.append(
            f"- Orphaned pages (no inbound links): {', '.join(orphans)}"
        )
    entry_lines += ["", "---", ""]

    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")
        # Insert new entry at the top, below the header
        if existing.startswith("# Wiki Log"):
            header, _, rest = existing.partition("\n\n")
            content = header + "\n\n" + "\n".join(entry_lines) + rest
        else:
            content = "\n".join(entry_lines) + existing
    else:
        content = "# Wiki Log\n\n" + "\n".join(entry_lines)

    log_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_graph():
    if not INGESTION_LOG.exists():
        print("No ingestion log found. Run src/ingest.py first.")
        return

    log = json.loads(INGESTION_LOG.read_text())
    if not log:
        print("Nothing indexed yet. Run src/ingest.py first.")
        return

    collection = get_collection()
    WIKI_DIR.mkdir(exist_ok=True)

    # Only generate pages for files inside raw/ that still exist on disk
    indexed_files = [
        Path(p) for p in log
        if Path(p).is_relative_to(RAW_DIR) and Path(p).exists()
    ]

    if not indexed_files:
        print("No raw/ documents found in index. Run src/ingest.py first.")
        return

    print(f"Computing candidates for {len(indexed_files)} documents...")

    # Phase 1: collect metadata + candidate links for every document
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
            "source_type": meta_result["metadatas"][0].get("source_type", "unknown"),
        }
        all_candidates[filename] = _find_candidates(collection, filename)

    # Phase 2: mutual top-K filter
    print("Applying mutual top-K filter...")
    mutual_links = _apply_mutual_filter(all_candidates)
    total_links = sum(len(v) for v in mutual_links.values())
    print(f"  {total_links} mutual links across {len(indexed_files)} documents")

    # Phase 2b: detect orphans (pages with no inbound links)
    has_inbound = {
        other["filename"]
        for links in mutual_links.values()
        for other in links
    }
    orphans = sorted(
        _note_title(f) for f in mutual_links if f not in has_inbound
    )
    if orphans:
        print(f"  {len(orphans)} orphaned page(s) (no inbound links)")

    # Phase 2c: delete stale wiki pages for files no longer in the index
    expected_stems = {_note_title(f) for f in mutual_links}
    stale_deleted: list[str] = []
    for existing_page in WIKI_DIR.rglob("*.md"):
        if existing_page.name in ("index.md", "log.md"):
            continue
        if existing_page.stem not in expected_stems:
            existing_page.unlink()
            stale_deleted.append(existing_page.name)
            print(f"  Deleted stale page: {existing_page.relative_to(WIKI_DIR)}")

    # Phase 3: write wiki pages (with excerpts)
    docs_by_course: dict[str, list[dict]] = defaultdict(list)

    for filename, related in mutual_links.items():
        meta = doc_meta.get(filename, {})
        course = meta.get("course", "Unknown")
        chunk_count = meta.get("chunk_count", 0)
        excerpts = _get_excerpts(collection, filename)

        course_dir = WIKI_DIR / course
        course_dir.mkdir(exist_ok=True)
        page_path = course_dir / f"{_note_title(filename)}.md"
        _write_page(page_path, filename, course, chunk_count, related, excerpts,
                    source_type=meta.get("source_type", "unknown"))

        # collect preview for index (first sentence of first excerpt)
        preview = _first_sentence(excerpts[0]) if excerpts else ""
        docs_by_course[course].append({
            "title": _note_title(filename),
            "chunks": chunk_count,
            "preview": preview,
        })

        print(f"  {_note_title(filename)}.md -> {len(related)} links")

    # Phase 4: regenerate wiki/index.md
    _write_index(WIKI_DIR, docs_by_course)
    print(f"  wiki/index.md updated ({len(indexed_files)} entries)")

    # Phase 5: append to wiki/log.md
    _append_log(WIKI_DIR, len(indexed_files), total_links, stale_deleted, orphans)
    print(f"  wiki/log.md updated")

    print(f"\nDone. {len(indexed_files)} wiki pages written to wiki/")
    print("Open Obsidian and switch to Graph View to see connections.")


if __name__ == "__main__":
    run_graph()
