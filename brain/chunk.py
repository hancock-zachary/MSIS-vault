from pathlib import Path
import fitz  # pymupdf
import tiktoken
from brain.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def extract_pages(pdf_path: Path, course: str) -> list[dict]:
    with fitz.open(str(pdf_path)) as doc:
        outline = {p: title for title, p in _extract_outline(doc)}
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if not text:
                continue
            pages.append({
                "course": course,
                "filename": pdf_path.name,
                "page": i + 1,
                "slide_title": outline.get(i, ""),
                "text": text,
            })
    return pages


def _extract_outline(doc) -> list[tuple]:
    try:
        return [(entry[0], entry[2]) for entry in doc.get_toc()]
    except Exception:
        return []


def chunk_page(page: dict) -> list[dict]:
    tokens = _enc.encode(page["text"])
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_text = _enc.decode(tokens[start:end])
        chunk_id = f"{page['course']}_{page['filename']}_p{page['page']}_c{idx}"
        chunks.append({
            "id": chunk_id,
            "course": page["course"],
            "filename": page["filename"],
            "page": page["page"],
            "slide_title": page["slide_title"],
            "chunk_index": idx,
            "text": chunk_text,
        })
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP_TOKENS
        idx += 1
    return chunks


def build_chunks_from_pdf(pdf_path: Path, course: str) -> list[dict]:
    pages = extract_pages(pdf_path, course)
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page))
    return chunks
