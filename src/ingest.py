import json
from pathlib import Path
from src.config import (
    RAW_DIR, INGESTION_LOG, BM25_PATH, PURGE_ABORT_FRACTION,
    SOURCE_TYPE_KEYWORDS, SUPPORTED_EXTENSIONS,
)
from src.chunk import build_chunks_from_file, enrich_for_embedding, is_quality_text
from src.embed import embed_batch
from src.index import get_collection, upsert_chunks, rebuild_bm25_from_collection


def load_log(log_path: Path) -> dict:
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def log_indexed(log: dict, pdf_path: Path, log_path: Path):
    log[str(pdf_path)] = "done"
    log_path.write_text(json.dumps(log, indent=2))


def purge_deleted_files(log: dict, log_path: Path, allow_purge: bool = False):
    """Drop index entries for files that no longer exist on disk.

    This only ever removes chunks from ChromaDB and entries from the log — it
    never touches files under raw/. Source documents are never deleted here.

    Safety guard: if more than PURGE_ABORT_FRACTION of indexed files are
    missing at once, that usually means raw/ was moved or reorganized rather
    than files being intentionally deleted. In that case we abort without
    changing anything, so a botched reorg can't silently wipe the index.
    Re-run with --purge to override.
    """
    missing = [p for p in list(log) if not Path(p).exists()]
    if not missing:
        return

    fraction = len(missing) / len(log)
    if fraction > PURGE_ABORT_FRACTION and not allow_purge:
        print(f"\n*** PURGE GUARD: {len(missing)} of {len(log)} indexed files "
              f"({fraction:.0%}) are missing from disk. ***")
        print("This usually means raw/ was moved or reorganized, not that these")
        print("files should be dropped. Refusing to purge the index automatically.")
        print("(This only affects the search index — your raw/ documents are not touched.)")
        for p in missing[:30]:
            print(f"  would drop index entries for: {p}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")
        print("\nIf this is intentional, re-run: uv run python src/ingest.py --purge")
        raise SystemExit(1)

    collection = get_collection()
    for file_path_str in missing:
        path = Path(file_path_str)
        filename = path.name
        try:
            # Scope deletion to course + filename so a same-named file in
            # another course is never collaterally purged.
            where = {"$and": [{"filename": filename}, {"course": _course_from_path(path)}]}
        except ValueError:
            where = {"filename": filename}
        existing = collection.get(where=where, include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  Removed {len(existing['ids'])} index chunk(s) for file no longer on disk: {filename}")
        del log[file_path_str]

    log_path.write_text(json.dumps(log, indent=2))
    print(f"Purged index entries for {len(missing)} file(s) missing from disk "
          f"(raw/ documents were NOT touched).")


def find_unindexed_files(root: Path, log: dict) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and str(p) not in log
    ]


def _course_from_path(file_path: Path) -> str:
    """Infer course name from the first subfolder under raw/.

    raw/IS 6410/slides/file.pdf  →  "IS 6410"
    raw/IS 6410/file.pdf         →  "IS 6410"
    raw/file.pdf                 →  "General"
    """
    parts = file_path.relative_to(RAW_DIR).parts
    return parts[0] if len(parts) > 1 else "General"


def _source_type_from_path(file_path: Path) -> str:
    """Infer document authority tier from the subfolder under raw/<Course>/.

    raw/IS 6410/slides/week1.pdf      →  "slides"
    raw/IS 6410/transcripts/t1.txt    →  "transcript"
    raw/IS 6410/week1.pdf             →  "unknown"  (no type subfolder)
    """
    parts = file_path.relative_to(RAW_DIR).parts
    if len(parts) < 3:
        return "unknown"
    folder = parts[1].lower()
    for keyword, source_type in SOURCE_TYPE_KEYWORDS.items():
        if keyword in folder:
            return source_type
    return "unknown"


def run_ingestion(allow_purge: bool = False):
    log = load_log(INGESTION_LOG)
    purge_deleted_files(log, INGESTION_LOG, allow_purge=allow_purge)
    files = find_unindexed_files(RAW_DIR, log)
    if not files:
        print("Nothing to index.")

    collection = get_collection()

    for file_path in files:
        course = _course_from_path(file_path)
        print(f"Indexing {file_path.name} ({course})...")
        try:
            chunks = build_chunks_from_file(file_path, course)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue
        if not chunks:
            print(f"  WARNING: no extractable text in {file_path.name}, skipping.")
            continue
        before = len(chunks)
        chunks = [c for c in chunks if c.get("is_stub") or is_quality_text(c["text"])]
        dropped = before - len(chunks)
        if dropped:
            print(f"  Dropped {dropped} garbled chunk(s) (image/diagram text).")
        if not chunks:
            print(f"  WARNING: all chunks were garbled in {file_path.name}, skipping.")
            continue
        source_type = _source_type_from_path(file_path)
        for c in chunks:
            c["source_type"] = source_type
        vectors = embed_batch([enrich_for_embedding(c) for c in chunks])
        upsert_chunks(collection, chunks, vectors)
        log_indexed(log, file_path, INGESTION_LOG)
        print(f"  {len(chunks)} chunks indexed.")

    # Always rebuild BM25 from ChromaDB — the single source of truth — so the
    # two stores stay consistent even after purges, crashes, or no-op runs.
    total = rebuild_bm25_from_collection(collection, BM25_PATH)
    print(f"Done. BM25 index rebuilt from ChromaDB with {total} total chunks.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index new/changed files under raw/.")
    parser.add_argument(
        "--purge", action="store_true",
        help="allow purging index entries even when a large fraction of indexed "
             "files are missing from disk (overrides the purge safety guard)",
    )
    args = parser.parse_args()
    run_ingestion(allow_purge=args.purge)
