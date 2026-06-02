from pathlib import Path
import fitz  # pymupdf
import tiktoken
from brain.config import (
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
    SLIDE_PAGE_TOKEN_THRESHOLD, SLIDE_PAGES_PER_CHUNK,
    SUPPORTED_EXTENSIONS,
)

_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Format-specific extractors — each returns a list of page dicts
# ---------------------------------------------------------------------------

def _extract_pages_pdf(pdf_path: Path, course: str) -> list[dict]:
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


def _extract_pages_docx(path: Path, course: str) -> list[dict]:
    """Extract text from .docx files paragraph by paragraph.

    Paragraphs are joined into one body of text and treated as a single
    logical document (not slides), so the standard overlapping-window
    chunker handles splitting.
    """
    from docx import Document  # deferred — only needed for .docx files
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not full_text.strip():
        return []
    return [{"course": course, "filename": path.name, "page": 1,
             "slide_title": "", "text": full_text}]


def _extract_pages_text(path: Path, course: str) -> list[dict]:
    """Extract text from .txt and .md files.

    Reads the file as-is. Markdown syntax is preserved — embedding models
    handle it fine, and stripping it risks losing structural meaning.
    YAML frontmatter (common in Obsidian notes) is stripped.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    # strip YAML frontmatter if present (--- ... ---)
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            raw = raw[end + 3:].strip()
    if not raw:
        return []
    return [{"course": course, "filename": path.name, "page": 1,
             "slide_title": "", "text": raw}]


def extract_pages(path: Path, course: str) -> list[dict]:
    """Dispatch to the correct extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pages_pdf(path, course)
    if suffix == ".docx":
        return _extract_pages_docx(path, course)
    if suffix in (".txt", ".md"):
        return _extract_pages_text(path, course)
    if suffix == ".doc":
        raise ValueError(
            f"{path.name} is an old .doc file. Please save it as .docx and re-ingest."
        )
    raise ValueError(f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _avg_tokens_per_page(pages: list[dict]) -> float:
    if not pages:
        return 0.0
    return sum(len(_enc.encode(p["text"])) for p in pages) / len(pages)


def is_slide_deck(pages: list[dict]) -> bool:
    """True when average tokens per page is below the slide threshold."""
    return _avg_tokens_per_page(pages) < SLIDE_PAGE_TOKEN_THRESHOLD


def chunk_slides(pages: list[dict]) -> list[dict]:
    """Group consecutive pages into chunks for slide decks."""
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


def build_chunks_from_file(path: Path, course: str) -> list[dict]:
    """Extract and chunk any supported file type."""
    pages = extract_pages(path, course)
    if not pages:
        return []
    if is_slide_deck(pages):
        return chunk_slides(pages)
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page))
    return chunks
