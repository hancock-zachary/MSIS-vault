from collections import defaultdict
from pathlib import Path
import re
import fitz  # pymupdf
import tiktoken
import numpy as np
from src.config import (
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
    SLIDE_PAGE_TOKEN_THRESHOLD, SLIDE_PAGES_PER_CHUNK,
    SUPPORTED_EXTENSIONS, BOILERPLATE_PAGE_RATIO, BOILERPLATE_MIN_PAGES,
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
                "slide_title": outline.get(i + 1, ""),  # outline keys are 1-based pages
                "text": text,
            })
    return pages


def _extract_outline(doc) -> list[tuple]:
    """Return (title, page) tuples from the PDF table of contents.

    get_toc() entries are [level, title, page] with 1-based page numbers.
    """
    try:
        return [(entry[1], entry[2]) for entry in doc.get_toc()]
    except Exception:
        return []


def _extract_pages_docx(path: Path, course: str) -> list[dict]:
    """Extract text from .docx files, returning one page per heading section.

    Paragraphs with Heading1, Heading2, or Heading3 styles mark section
    boundaries. If no headings are found, returns a single page with all text.
    """
    from docx import Document
    doc = Document(str(path))

    HEADING_STYLES = {"heading 1", "heading 2", "heading 3"}
    sections = []
    current_heading = ""
    current_body: list[str] = []

    for para in doc.paragraphs:
        style_name = para.style.name.lower() if para.style else ""
        text = para.text.strip()
        if not text:
            continue
        if style_name in HEADING_STYLES:
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = text
            current_body = []
        else:
            current_body.append(text)

    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body)))

    if not sections:
        return []

    # If only one section with no heading (no headings found), return as single page
    if len(sections) == 1 and not sections[0][0]:
        full_text = sections[0][1]
        if not full_text.strip():
            return []
        return [{"course": course, "filename": path.name, "page": 1,
                 "slide_title": "", "text": full_text}]

    return [
        {
            "course": course,
            "filename": path.name,
            "page": i + 1,
            "slide_title": heading,
            "text": f"{heading}\n\n{body}".strip() if heading else body,
        }
        for i, (heading, body) in enumerate(sections)
        if heading or body
    ]


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


def is_quality_text(text: str) -> bool:
    """Return False if text looks like garbled image/diagram extraction."""
    words = text.split()
    if len(words) < 10:
        return False
    single_char = sum(1 for w in words if len(re.sub(r"[^a-zA-Z]", "", w)) <= 1)
    if single_char / len(words) > 0.3:
        return False
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(len(text), 1) < 0.4:
        return False
    return True


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
# Boilerplate stripping
# ---------------------------------------------------------------------------

def _strip_boilerplate(pages: list[dict]) -> list[dict]:
    """Remove lines that repeat across too many pages (headers, footers, watermarks).

    A line is boilerplate if it appears verbatim (whitespace-normalised) on more
    than BOILERPLATE_PAGE_RATIO of all pages in the document. Typical culprits:
    course title, university name, professor copyright, slide deck branding.

    Stripping happens before chunking so boilerplate doesn't pollute embeddings
    or create false similarity matches between unrelated documents.

    Skipped when the document has fewer than BOILERPLATE_MIN_PAGES pages — too
    little signal to distinguish boilerplate from legitimate repeated content.
    """
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return pages

    # Count how many distinct pages each normalised line appears on
    line_page_count: dict[str, int] = defaultdict(int)
    for page in pages:
        seen_on_this_page: set[str] = set()
        for raw_line in page["text"].splitlines():
            norm = " ".join(raw_line.split())
            if norm and norm not in seen_on_this_page:
                line_page_count[norm] += 1
                seen_on_this_page.add(norm)

    threshold = len(pages) * BOILERPLATE_PAGE_RATIO
    boilerplate = {line for line, count in line_page_count.items() if count >= threshold}

    if not boilerplate:
        return pages

    cleaned = []
    for page in pages:
        filtered_lines = [
            line for line in page["text"].splitlines()
            if " ".join(line.split()) not in boilerplate
        ]
        cleaned.append({**page, "text": "\n".join(filtered_lines).strip()})
    return cleaned


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
            "page_start": first["page"],
            "page_end": group[-1]["page"],
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on punctuation followed by whitespace and capital."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in parts if s.strip()]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0.0:
        return 0.0
    return float(np.dot(va, vb) / norm)


def chunk_semantic(page: dict, embed_fn) -> list[dict]:
    """Split a page at semantic boundaries using embedding similarity.

    Embeds each sentence, finds consecutive pairs whose cosine similarity
    drops below SEMANTIC_SPLIT_THRESHOLD, and splits there. Falls back to
    overlapping window chunking if too few sentences or no splits found.
    """
    from src.config import SEMANTIC_SPLIT_THRESHOLD, CHUNK_SIZE_TOKENS

    sentences = _split_sentences(page["text"])
    if len(sentences) <= 1:
        chunks = chunk_page(page)
        for c in chunks:
            c["strategy"] = "semantic"
            c["is_stub"] = False
        return chunks

    vectors = embed_fn(sentences)
    assert len(vectors) == len(sentences), (
        f"embed_fn returned {len(vectors)} vectors for {len(sentences)} sentences"
    )

    split_points = []
    for i in range(len(vectors) - 1):
        sim = _cosine_sim(vectors[i], vectors[i + 1])
        if sim < SEMANTIC_SPLIT_THRESHOLD:
            split_points.append(i + 1)

    if not split_points:
        chunks = chunk_page(page)
        for c in chunks:
            c["strategy"] = "semantic"
            c["is_stub"] = False
        return chunks

    boundaries = [0] + split_points + [len(sentences)]
    segments = [
        " ".join(sentences[boundaries[i]:boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
        if " ".join(sentences[boundaries[i]:boundaries[i + 1]]).strip()
    ]

    chunks = []
    for idx, seg_text in enumerate(segments):
        seg_page = dict(page)
        seg_page["text"] = seg_text
        if len(_enc.encode(seg_text)) > CHUNK_SIZE_TOKENS:
            sub_chunks = chunk_page(seg_page)
            for c in sub_chunks:
                new_index = idx * 100 + c["chunk_index"]
                c["chunk_index"] = new_index
                c["id"] = f"{page['course']}_{page['filename']}_p{page['page']}_c{new_index}"
                c["strategy"] = "semantic"
                c["is_stub"] = False
            chunks.extend(sub_chunks)
        else:
            chunk_id = f"{page['course']}_{page['filename']}_p{page['page']}_c{idx}"
            if len(_enc.encode(seg_text)) >= 10:
                chunks.append({
                    "id": chunk_id,
                    "course": page["course"],
                    "filename": page["filename"],
                    "page": page["page"],
                    "slide_title": page["slide_title"],
                    "chunk_index": idx,
                    "text": seg_text,
                    "strategy": "semantic",
                    "is_stub": False,
                })

    if not chunks:
        chunks = chunk_page(page)
        for c in chunks:
            c["strategy"] = "semantic"
            c["is_stub"] = False

    # Renumber all chunks sequentially to guarantee unique IDs regardless of
    # how many sub-chunks oversized segments produced.
    for final_idx, c in enumerate(chunks):
        c["id"] = f"{page['course']}_{page['filename']}_p{page['page']}_c{final_idx}"
        c["chunk_index"] = final_idx

    return chunks


def chunk_structured(pages: list[dict]) -> list[dict]:
    """Split documents on heading boundaries for MD, DOCX, and outlined PDFs.

    For MD: splits on lines matching ^#{1,3} heading patterns.
    For DOCX/PDF: pages already contain one section per page (from extraction),
    so each page becomes one chunk (with window fallback if too large).
    """
    chunks = []

    for page in pages:
        ext = Path(page["filename"]).suffix.lower()

        if ext == ".md":
            heading_re = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
            text = page["text"]
            matches = list(heading_re.finditer(text))

            if not matches:
                sub_chunks = chunk_page(page)
                for c in sub_chunks:
                    c["strategy"] = "structured"
                    c["is_stub"] = False
                chunks.extend(sub_chunks)
                continue

            sections = []
            for i, match in enumerate(matches):
                heading_text = match.group(2).strip()
                body_start = match.end()
                body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[body_start:body_end].strip()
                section_text = f"{heading_text}\n\n{body}".strip() if body else heading_text
                sections.append((heading_text, section_text))

            for idx, (heading, section_text) in enumerate(sections):
                section_page = dict(page)
                section_page["text"] = section_text
                section_page["slide_title"] = heading

                if len(_enc.encode(section_text)) > CHUNK_SIZE_TOKENS:
                    sub_chunks = chunk_page(section_page)
                    for c in sub_chunks:
                        new_index = idx * 100 + c["chunk_index"]
                        c["chunk_index"] = new_index
                        c["id"] = f"{page['course']}_{page['filename']}_p{page['page']}_c{new_index}"
                        c["strategy"] = "structured"
                        c["is_stub"] = False
                    chunks.extend(sub_chunks)
                else:
                    chunk_id = f"{page['course']}_{page['filename']}_p{page['page']}_c{idx}"
                    chunks.append({
                        "id": chunk_id,
                        "course": page["course"],
                        "filename": page["filename"],
                        "page": page["page"],
                        "slide_title": heading,
                        "chunk_index": idx,
                        "text": section_text,
                        "strategy": "structured",
                        "is_stub": False,
                    })

        else:
            # DOCX or PDF with outline: each page is already a section
            sub_chunks = chunk_page(page)
            for c in sub_chunks:
                c["strategy"] = "structured"
                c["is_stub"] = False
            chunks.extend(sub_chunks)

    return chunks


def enrich_for_embedding(chunk: dict) -> str:
    """Build the text sent to the embedding model: a context header (course,
    document, slide title) prepended to the chunk text, so the vector carries
    surrounding context the chunk text alone lacks. The stored chunk text
    stays clean — enrichment exists only in the embedding input."""
    parts = [
        chunk.get("course", ""),
        Path(chunk["filename"]).stem,
        chunk.get("slide_title", ""),
    ]
    header = " - ".join(p for p in parts if p)
    return f"{header}\n{chunk['text']}"


def build_chunks_from_file(path: Path, course: str) -> list[dict]:
    """Extract and chunk any supported file type using the signal-based router.

    Returns chunks with 'strategy' and 'is_stub' metadata fields on every chunk.
    Stub chunks (is_stub=True) are salvaged title text from garbled image pages.
    """
    from src.router import build_profile, route_and_chunk
    from src.embed import embed_batch

    pages = extract_pages(path, course)
    if not pages:
        return []
    pages = _strip_boilerplate(pages)
    if not pages:
        return []

    profile = build_profile(path, pages)
    return route_and_chunk(profile, pages, embed_fn=embed_batch)
