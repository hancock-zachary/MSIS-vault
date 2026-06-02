from pathlib import Path
import fitz  # pymupdf
import tiktoken
from brain.config import (
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
    SLIDE_PAGE_TOKEN_THRESHOLD, SLIDE_PAGES_PER_CHUNK,
)

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


def _avg_tokens_per_page(pages: list[dict]) -> float:
    if not pages:
        return 0.0
    return sum(len(_enc.encode(p["text"])) for p in pages) / len(pages)


def is_slide_deck(pages: list[dict]) -> bool:
    """True when average tokens per page is below the slide threshold."""
    return _avg_tokens_per_page(pages) < SLIDE_PAGE_TOKEN_THRESHOLD


def chunk_slides(pages: list[dict]) -> list[dict]:
    """Group consecutive pages into chunks for slide decks.

    Grouping preserves the narrative arc of a lecture section and produces
    embeddings rich enough to connect across course boundaries.
    """
    chunks = []
    for group_idx, i in enumerate(range(0, len(pages), SLIDE_PAGES_PER_CHUNK)):
        group = pages[i:i + SLIDE_PAGES_PER_CHUNK]
        combined_text = "\n\n".join(p["text"] for p in group)
        first = group[0]
        chunk_id = f"{first['course']}_{first['filename']}_p{first['page']}_c{group_idx}"
        chunks.append({
            "id": chunk_id,
            "course": first["course"],
            "filename": first["filename"],
            "page": first["page"],
            "slide_title": first["slide_title"],
            "chunk_index": group_idx,
            "text": combined_text,
        })
    return chunks


def chunk_page(page: dict) -> list[dict]:
    """Split a single dense-text page into overlapping token windows."""
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
    if not pages:
        return []
    if is_slide_deck(pages):
        return chunk_slides(pages)
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page))
    return chunks
