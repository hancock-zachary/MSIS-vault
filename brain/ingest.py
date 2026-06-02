import json
import re
from pathlib import Path
from brain.config import COURSES_DIR, INGESTION_LOG, BM25_PATH
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


def _is_quality_chunk(text: str) -> bool:
    """Return False if text looks like garbled image extraction.

    Cartoons, diagrams, and scanned images produce chunks with a high ratio of
    single-character words and non-alphabetic noise. We discard those rather
    than polluting the index with meaningless vectors.
    """
    words = text.split()
    if len(words) < 10:
        return False  # too short to be meaningful
    single_char = sum(1 for w in words if len(re.sub(r"[^a-zA-Z]", "", w)) <= 1)
    if single_char / len(words) > 0.3:
        return False  # more than 30% single-character words → garbled
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(len(text), 1) < 0.4:
        return False  # less than 40% alphabetic characters → noise
    return True


def find_unindexed_pdfs(root: Path, log: dict) -> list[Path]:
    return [p for p in root.rglob("*.pdf") if str(p) not in log]


def _course_from_path(pdf_path: Path) -> str:
    """Infer course name from courses/ folder structure: courses/<course>/..."""
    return pdf_path.relative_to(COURSES_DIR).parts[0]


def run_ingestion():
    log = load_log(INGESTION_LOG)
    pdfs = find_unindexed_pdfs(COURSES_DIR, log)
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
        before = len(chunks)
        chunks = [c for c in chunks if _is_quality_chunk(c["text"])]
        dropped = before - len(chunks)
        if dropped:
            print(f"  Dropped {dropped} garbled chunk(s) (image/diagram text).")
        if not chunks:
            print(f"  WARNING: all chunks were garbled in {pdf_path.name}, skipping.")
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
