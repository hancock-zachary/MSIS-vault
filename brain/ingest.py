import json
from pathlib import Path
from brain.config import VAULT_ROOT, INGESTION_LOG, BM25_PATH
from brain.chunk import build_chunks_from_pdf
from brain.embed import embed_batch
from brain.index import get_collection, upsert_chunks, build_bm25, load_bm25


def load_log(log_path: Path) -> dict:
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def log_indexed(log: dict, pdf_path: Path, log_path: Path):
    log[str(pdf_path)] = "done"
    log_path.write_text(json.dumps(log, indent=2))


def find_unindexed_pdfs(root: Path, log: dict) -> list[Path]:
    return [p for p in root.rglob("*.pdf") if str(p) not in log]


def _course_from_path(pdf_path: Path) -> str:
    """Infer course name from parent folder structure."""
    parts = pdf_path.relative_to(VAULT_ROOT).parts
    return parts[1] if len(parts) > 2 else parts[0]


def run_ingestion():
    log = load_log(INGESTION_LOG)
    pdfs = find_unindexed_pdfs(VAULT_ROOT, log)
    if not pdfs:
        print("Nothing to index.")
        return

    collection = get_collection()

    # Load existing BM25 chunks to append to
    if BM25_PATH.exists():
        _, _, existing_chunks = load_bm25(BM25_PATH)  # (index, tokenized_corpus, chunks)
    else:
        existing_chunks = []

    all_new_chunks = []
    for pdf_path in pdfs:
        course = _course_from_path(pdf_path)
        print(f"Indexing {pdf_path.name} ({course})...")
        chunks = build_chunks_from_pdf(pdf_path, course)
        if not chunks:
            print(f"  WARNING: no extractable text in {pdf_path.name}, skipping.")
            continue
        vectors = embed_batch([c["text"] for c in chunks])
        upsert_chunks(collection, chunks, vectors)
        all_new_chunks.extend(chunks)
        log_indexed(log, pdf_path, INGESTION_LOG)
        print(f"  {len(chunks)} chunks indexed.")

    build_bm25(existing_chunks + all_new_chunks, BM25_PATH)
    print(f"Done. BM25 index rebuilt with {len(existing_chunks) + len(all_new_chunks)} total chunks.")


if __name__ == "__main__":
    run_ingestion()
