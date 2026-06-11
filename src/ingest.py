import json
from pathlib import Path
from src.config import RAW_DIR, INGESTION_LOG, BM25_PATH, SUPPORTED_EXTENSIONS
from src.chunk import build_chunks_from_file, enrich_for_embedding, is_quality_text
from src.embed import embed_batch
from src.index import get_collection, upsert_chunks, build_bm25, load_bm25


def load_log(log_path: Path) -> dict:
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def log_indexed(log: dict, pdf_path: Path, log_path: Path):
    log[str(pdf_path)] = "done"
    log_path.write_text(json.dumps(log, indent=2))


def purge_deleted_files(log: dict, log_path: Path):
    """Remove files that no longer exist from the log and ChromaDB collection."""
    missing = [p for p in list(log) if not Path(p).exists()]
    if not missing:
        return

    collection = get_collection()
    for file_path_str in missing:
        filename = Path(file_path_str).name
        existing = collection.get(where={"filename": filename}, include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  Removed {len(existing['ids'])} chunks for deleted file: {filename}")
        del log[file_path_str]

    log_path.write_text(json.dumps(log, indent=2))
    print(f"Purged {len(missing)} deleted file(s) from index.")

    # Rebuild BM25 without the deleted files' chunks
    if BM25_PATH.exists():
        _, _, existing_chunks = load_bm25(BM25_PATH)
        deleted_filenames = {Path(p).name for p in missing}
        surviving_chunks = [c for c in existing_chunks if c.get("filename") not in deleted_filenames]
        build_bm25(surviving_chunks, BM25_PATH)
        print(f"  BM25 rebuilt with {len(surviving_chunks)} remaining chunks.")


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


def run_ingestion():
    log = load_log(INGESTION_LOG)
    purge_deleted_files(log, INGESTION_LOG)
    files = find_unindexed_files(RAW_DIR, log)
    if not files:
        print("Nothing to index.")
        return

    collection = get_collection()

    # Load existing BM25 chunks to append to
    if BM25_PATH.exists():
        _, _, existing_chunks = load_bm25(BM25_PATH)  # (index, tokenized_corpus, chunks)
    else:
        existing_chunks = []

    all_new_chunks = []
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
        vectors = embed_batch([enrich_for_embedding(c) for c in chunks])
        upsert_chunks(collection, chunks, vectors)
        all_new_chunks.extend(chunks)
        log_indexed(log, file_path, INGESTION_LOG)
        print(f"  {len(chunks)} chunks indexed.")

    build_bm25(existing_chunks + all_new_chunks, BM25_PATH)
    print(f"Done. BM25 index rebuilt with {len(existing_chunks) + len(all_new_chunks)} total chunks.")


if __name__ == "__main__":
    run_ingestion()
