# Smart Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-strategy chunking pipeline with a signal-based router that selects the optimal strategy per document, attaches `ChunkingProfile` metadata to every chunk, and salvages title text from garbled image pages as flagged stub chunks.

**Architecture:** A new `src/router.py` owns `ChunkingProfile`, signal detection, routing logic, and the salvage pass. `src/chunk.py` gains `chunk_semantic` and `chunk_structured` strategies and loses routing responsibility. `build_chunks_from_file` integrates the router internally so `ingest.py` changes are minimal.

**Tech Stack:** Python 3.11+, numpy (already in env), tiktoken, pymupdf, python-docx, sentence-transformers (embed via src.embed)

---

## File Structure

```
src/
  router.py              ← NEW: ChunkingProfile, build_profile, route_and_chunk, salvage_pass
  chunk.py               ← MODIFIED: add chunk_semantic, chunk_structured, is_quality_text; refactor build_chunks_from_file
  ingest.py              ← MODIFIED: import is_quality_text from chunk, update quality filter to pass stubs
  config.py              ← MODIFIED: add SEMANTIC_SPLIT_THRESHOLD, MIN_STUB_TOKENS
  tests/
    test_router.py        ← NEW
    test_chunk.py         ← MODIFIED: new tests for chunk_semantic, chunk_structured, metadata fields
    test_ingest.py        ← MODIFIED: update quality filter test
```

---

## Task 1: Config constants

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add constants**

Open `src/config.py` and add after `CHUNK_OVERLAP_TOKENS`:

```python
SEMANTIC_SPLIT_THRESHOLD = 0.4  # cosine similarity drop below this triggers a semantic split
MIN_STUB_TOKENS = 3              # minimum meaningful words for a garbled-page title to become a stub
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.config import SEMANTIC_SPLIT_THRESHOLD, MIN_STUB_TOKENS; print(SEMANTIC_SPLIT_THRESHOLD, MIN_STUB_TOKENS)"`
Expected: `0.4 3`

- [ ] **Step 3: Commit**

```
git add src/config.py
git commit -m "feat: add SEMANTIC_SPLIT_THRESHOLD and MIN_STUB_TOKENS to config"
```

---

## Task 2: Move quality check to chunk.py

The `_is_quality_chunk` function currently lives in `ingest.py` but is needed by `router.py` too. Move it to `chunk.py` as `is_quality_text` so both modules can import it cleanly.

**Files:**
- Modify: `src/chunk.py`
- Modify: `src/ingest.py`
- Modify: `src/tests/test_ingest.py`
- Modify: `src/tests/test_chunk.py`

- [ ] **Step 1: Write failing test in test_chunk.py**

Add to `src/tests/test_chunk.py`:

```python
from src.chunk import is_quality_text

def test_is_quality_text_accepts_normal_text():
    text = "The five Scrum events are Sprint Planning, Daily Scrum, Sprint Review, Sprint Retrospective, and the Sprint itself."
    assert is_quality_text(text) is True

def test_is_quality_text_rejects_garbled():
    garbled = "E 8 c I M O O 75 2 1 S xa S e U n E fl o ij ri O CM o COrAPLICATED THAT NO ONE KNOWS WHO OOES WHAT"
    assert is_quality_text(garbled) is False

def test_is_quality_text_rejects_too_short():
    assert is_quality_text("too short") is False
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest src/tests/test_chunk.py::test_is_quality_text_accepts_normal_text -v`
Expected: FAIL — `ImportError: cannot import name 'is_quality_text'`

- [ ] **Step 3: Add is_quality_text to chunk.py**

Add at the top of `src/chunk.py` (after existing imports):

```python
import re
```

Add this function before `extract_pages`:

```python
def is_quality_text(text: str) -> bool:
    """Return False if text looks like garbled image/diagram extraction.

    High ratio of single-character words or low alphabetic character density
    indicates OCR noise from cartoons, diagrams, or scanned images.
    """
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
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `uv run pytest src/tests/test_chunk.py::test_is_quality_text_accepts_normal_text src/tests/test_chunk.py::test_is_quality_text_rejects_garbled src/tests/test_chunk.py::test_is_quality_text_rejects_too_short -v`
Expected: All 3 PASS

- [ ] **Step 5: Update ingest.py to import from chunk**

In `src/ingest.py`:

Change:
```python
import json
import re
from pathlib import Path
from src.config import RAW_DIR, INGESTION_LOG, BM25_PATH, SUPPORTED_EXTENSIONS
from src.chunk import build_chunks_from_file
```

To:
```python
import json
from pathlib import Path
from src.config import RAW_DIR, INGESTION_LOG, BM25_PATH, SUPPORTED_EXTENSIONS
from src.chunk import build_chunks_from_file, is_quality_text
```

Remove the entire `_is_quality_chunk` function from `ingest.py` (lines 21–37).

Replace every occurrence of `_is_quality_chunk` with `is_quality_text` in `ingest.py`:

```python
        before = len(chunks)
        chunks = [c for c in chunks if c.get("is_stub") or is_quality_text(c["text"])]
        dropped = before - len(chunks)
```

Note: `c.get("is_stub")` ensures stub chunks are never filtered out by the quality check.

- [ ] **Step 6: Update test_ingest.py**

In `src/tests/test_ingest.py`, change the import line:

```python
from src.ingest import find_unindexed_files, log_indexed, load_log, _course_from_path
from src.config import RAW_DIR
from src.chunk import is_quality_text
```

Remove the three `_is_quality_chunk` tests (they now live in `test_chunk.py`) and replace with:

```python
def test_is_quality_text_imported_correctly():
    # Verify the function is importable from chunk (used by ingest)
    assert callable(is_quality_text)
```

- [ ] **Step 7: Run full suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 8: Commit**

```
git add src/chunk.py src/ingest.py src/tests/test_chunk.py src/tests/test_ingest.py
git commit -m "refactor: move is_quality_text to chunk.py, stubs bypass quality filter"
```

---

## Task 3: ChunkingProfile and signal detection

**Files:**
- Create: `src/router.py`
- Create: `src/tests/test_router.py`

- [ ] **Step 1: Write failing tests**

Create `src/tests/test_router.py`:

```python
import pytest
from pathlib import Path
from src.router import ChunkingProfile, build_profile


def _make_pages(n: int, tokens_per_page: int, course="IS 6410", filename="test.pdf") -> list[dict]:
    text = "word " * tokens_per_page
    return [
        {"course": course, "filename": filename, "page": i + 1,
         "slide_title": "", "text": text}
        for i in range(n)
    ]


def test_build_profile_returns_dataclass(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.touch()
    pages = _make_pages(5, 50)
    profile = build_profile(pdf, pages)
    assert isinstance(profile, ChunkingProfile)
    assert profile.file_extension == ".pdf"
    assert profile.page_count == 5


def test_build_profile_routes_slides_for_short_pages(tmp_path):
    pdf = tmp_path / "slides.pdf"
    pdf.touch()
    pages = _make_pages(10, 30)  # 30 tokens/page << 150 threshold
    profile = build_profile(pdf, pages)
    assert profile.strategy == "slides"
    assert profile.avg_tokens_per_page < 150


def test_build_profile_routes_semantic_for_dense_pdf(tmp_path):
    pdf = tmp_path / "reading.pdf"
    pdf.touch()
    pages = _make_pages(5, 300)  # 300 tokens/page >> 150 threshold
    profile = build_profile(pdf, pages)
    assert profile.strategy == "semantic"


def test_build_profile_routes_structured_for_md(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Heading\n\nContent here.", encoding="utf-8")
    pages = [{"course": "IS 6410", "filename": "notes.md", "page": 1,
              "slide_title": "", "text": "# Heading\n\nContent here."}]
    profile = build_profile(md, pages)
    assert profile.strategy == "structured"
    assert profile.file_extension == ".md"


def test_build_profile_routes_structured_for_docx(tmp_path):
    docx = tmp_path / "report.docx"
    docx.touch()
    pages = _make_pages(1, 200, filename="report.docx")
    profile = build_profile(docx, pages)
    assert profile.strategy == "structured"


def test_build_profile_computes_garbled_ratio(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    pdf.touch()
    good_text = "word " * 50
    bad_text = "E 8 c I M O xa S e U n fl o ij ri O CM COrAPLICATED OOES WHAT single chars"
    pages = [
        {"course": "IS 6410", "filename": "mixed.pdf", "page": 1,
         "slide_title": "", "text": good_text},
        {"course": "IS 6410", "filename": "mixed.pdf", "page": 2,
         "slide_title": "", "text": bad_text},
    ]
    profile = build_profile(pdf, pages)
    assert profile.garbled_page_ratio == pytest.approx(0.5)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest src/tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.router'`

- [ ] **Step 3: Implement src/router.py (profile and signal detection only)**

Create `src/router.py`:

```python
"""
Signal-based chunking router.

Computes a ChunkingProfile from measurable document signals and routes each
document to the appropriate chunking strategy. Every decision is recorded in
the profile and attached as metadata to the resulting chunks.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from src.chunk import is_quality_text
from src.config import (
    SLIDE_PAGE_TOKEN_THRESHOLD,
)

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkingProfile:
    strategy: str              # "slides" | "semantic" | "structured" | "window"
    avg_tokens_per_page: float
    has_structure: bool        # headings or PDF outline detected
    garbled_page_ratio: float  # fraction of pages that failed quality check
    file_extension: str
    page_count: int


def _avg_tokens_per_page(pages: list[dict]) -> float:
    if not pages:
        return 0.0
    return sum(len(_enc.encode(p["text"])) for p in pages) / len(pages)


def _has_structure(path: Path, pages: list[dict]) -> bool:
    """True if the document has detectable heading structure."""
    ext = path.suffix.lower()
    if ext in (".docx", ".md"):
        return True  # always use structured strategy for these
    if ext == ".pdf":
        # PDF has structure if any page has a slide_title from the outline
        return any(p.get("slide_title", "") for p in pages)
    return False


def _compute_garbled_ratio(pages: list[dict]) -> float:
    if not pages:
        return 0.0
    garbled = sum(1 for p in pages if not is_quality_text(p["text"]))
    return garbled / len(pages)


def build_profile(path: Path, pages: list[dict]) -> ChunkingProfile:
    """Compute a ChunkingProfile from measurable document signals.

    Routing priority:
    1. .docx or .md → structured
    2. .pdf + avg tokens/page < SLIDE_PAGE_TOKEN_THRESHOLD → slides
    3. .pdf + outline/headings detected → structured
    4. everything else → semantic
    """
    ext = path.suffix.lower()
    avg_tokens = _avg_tokens_per_page(pages)
    has_struct = _has_structure(path, pages)
    garbled_ratio = _compute_garbled_ratio(pages)
    page_count = len(pages)

    if ext in (".docx", ".md"):
        strategy = "structured"
    elif ext == ".pdf" and avg_tokens < SLIDE_PAGE_TOKEN_THRESHOLD:
        strategy = "slides"
    elif ext == ".pdf" and has_struct:
        strategy = "structured"
    else:
        strategy = "semantic"

    return ChunkingProfile(
        strategy=strategy,
        avg_tokens_per_page=round(avg_tokens, 1),
        has_structure=has_struct,
        garbled_page_ratio=round(garbled_ratio, 3),
        file_extension=ext,
        page_count=page_count,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `uv run pytest src/tests/test_router.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```
git add src/router.py src/tests/test_router.py
git commit -m "feat: ChunkingProfile dataclass and signal-based routing in router.py"
```

---

## Task 4: chunk_semantic

**Files:**
- Modify: `src/chunk.py`
- Modify: `src/tests/test_chunk.py`

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_chunk.py`:

```python
from unittest.mock import patch
import numpy as np
from src.chunk import chunk_semantic

def _make_dense_page(text: str) -> dict:
    return {"course": "IS 6410", "filename": "reading.pdf", "page": 3,
            "slide_title": "", "text": text}

def test_chunk_semantic_returns_chunks_with_strategy():
    page = _make_dense_page("The sprint begins on Monday. The team plans the work. Daily scrums happen each morning.")
    fake_vectors = [[float(i)] * 768 for i in range(3)]

    def mock_embed(texts):
        return fake_vectors[:len(texts)]

    chunks = chunk_semantic(page, embed_fn=mock_embed)
    assert len(chunks) >= 1
    assert all(c["strategy"] == "semantic" for c in chunks)
    assert all(c["is_stub"] is False for c in chunks)

def test_chunk_semantic_falls_back_to_window_when_single_sentence():
    page = _make_dense_page("Just one sentence with no splits.")

    def mock_embed(texts):
        return [[0.1] * 768 for _ in texts]

    chunks = chunk_semantic(page, embed_fn=mock_embed)
    assert len(chunks) >= 1
    # fallback to window — strategy should still be set
    assert all("strategy" in c for c in chunks)

def test_chunk_semantic_splits_on_low_similarity():
    # Two clearly different topic sentences
    text = (
        "Scrum is an agile framework for managing software development. "
        "Supply chain management involves logistics and procurement."
    )
    page = _make_dense_page(text)

    call_count = [0]
    def mock_embed(texts):
        call_count[0] += len(texts)
        # Return orthogonal vectors to force a split
        vectors = []
        for i, t in enumerate(texts):
            v = [0.0] * 768
            v[i % 768] = 1.0
            vectors.append(v)
        return vectors

    chunks = chunk_semantic(page, embed_fn=mock_embed)
    assert call_count[0] > 0  # embed was called
    assert all(c["is_stub"] is False for c in chunks)
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest src/tests/test_chunk.py::test_chunk_semantic_returns_chunks_with_strategy -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_semantic'`

- [ ] **Step 3: Implement chunk_semantic in chunk.py**

Add these functions to `src/chunk.py` before `build_chunks_from_file`:

```python
import numpy as np


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

    Args:
        page: page dict with text, course, filename, page, slide_title fields.
        embed_fn: callable(list[str]) -> list[list[float]], the embedding provider.
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

    split_points = []
    for i in range(len(vectors) - 1):
        sim = _cosine_sim(vectors[i], vectors[i + 1])
        if sim < SEMANTIC_SPLIT_THRESHOLD:
            split_points.append(i + 1)

    if not split_points:
        # All sentences are similar — fall back to window
        chunks = chunk_page(page)
        for c in chunks:
            c["strategy"] = "semantic"
            c["is_stub"] = False
        return chunks

    # Build segments from split points
    boundaries = [0] + split_points + [len(sentences)]
    segments = [
        " ".join(sentences[boundaries[i]:boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
        if " ".join(sentences[boundaries[i]:boundaries[i + 1]]).strip()
    ]

    chunks = []
    for idx, seg_text in enumerate(segments):
        # Apply max token guardrail — split further if segment too large
        seg_page = dict(page)
        seg_page["text"] = seg_text
        if len(_enc.encode(seg_text)) > CHUNK_SIZE_TOKENS:
            sub_chunks = chunk_page(seg_page)
            for c in sub_chunks:
                c["chunk_index"] = idx * 100 + c["chunk_index"]
                c["strategy"] = "semantic"
                c["is_stub"] = False
            chunks.extend(sub_chunks)
        else:
            chunk_id = f"{page['course']}_{page['filename']}_p{page['page']}_c{idx}"
            if len(_enc.encode(seg_text)) >= 10:  # min token guardrail
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

    return chunks
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `uv run pytest src/tests/test_chunk.py::test_chunk_semantic_returns_chunks_with_strategy src/tests/test_chunk.py::test_chunk_semantic_falls_back_to_window_when_single_sentence src/tests/test_chunk.py::test_chunk_semantic_splits_on_low_similarity -v`
Expected: All 3 PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```
git add src/chunk.py src/tests/test_chunk.py
git commit -m "feat: add chunk_semantic with sentence-level cosine splitting"
```

---

## Task 5: chunk_structured + update DOCX extraction

**Files:**
- Modify: `src/chunk.py`
- Modify: `src/tests/test_chunk.py`

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_chunk.py`:

```python
from src.chunk import chunk_structured

def test_chunk_structured_splits_md_on_headings():
    page = {
        "course": "IS 6410", "filename": "notes.md", "page": 1,
        "slide_title": "", 
        "text": "# Sprint Planning\n\nSprint planning is the first event.\n\n# Daily Scrum\n\nThe daily scrum is 15 minutes."
    }
    chunks = chunk_structured([page])
    assert len(chunks) == 2
    assert "Sprint planning" in chunks[0]["text"]
    assert "daily scrum" in chunks[1]["text"]
    assert all(c["strategy"] == "structured" for c in chunks)
    assert all(c["is_stub"] is False for c in chunks)

def test_chunk_structured_no_headings_returns_window_chunks():
    page = {
        "course": "IS 6410", "filename": "notes.md", "page": 1,
        "slide_title": "",
        "text": "word " * 50  # no headings
    }
    chunks = chunk_structured([page])
    assert len(chunks) >= 1
    assert all(c["strategy"] == "structured" for c in chunks)

def test_chunk_structured_sets_slide_title_from_heading():
    page = {
        "course": "IS 6410", "filename": "notes.md", "page": 1,
        "slide_title": "",
        "text": "## Scrum Artifacts\n\nProduct backlog, sprint backlog, and increment."
    }
    chunks = chunk_structured([page])
    assert chunks[0]["slide_title"] == "Scrum Artifacts"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest src/tests/test_chunk.py::test_chunk_structured_splits_md_on_headings -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_structured'`

- [ ] **Step 3: Implement chunk_structured in chunk.py**

Add to `src/chunk.py` after `chunk_semantic`:

```python
def chunk_structured(pages: list[dict]) -> list[dict]:
    """Split documents on heading boundaries for MD, DOCX, and outlined PDFs.

    For MD: splits on lines matching ^#{1,3} heading patterns.
    For DOCX/PDF: pages already contain one section per page (from extraction),
    so each page becomes one chunk (with window fallback if too large).

    Args:
        pages: list of page dicts. For MD, typically a single page containing
               the full document text. For DOCX/PDF, one page per section.
    """
    chunks = []

    for page in pages:
        ext = Path(page["filename"]).suffix.lower()

        if ext == ".md":
            # Split on markdown heading lines
            heading_re = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
            text = page["text"]
            matches = list(heading_re.finditer(text))

            if not matches:
                # No headings — treat as one chunk
                sub_chunks = chunk_page(page)
                for c in sub_chunks:
                    c["strategy"] = "structured"
                    c["is_stub"] = False
                chunks.extend(sub_chunks)
                continue

            # Extract sections: heading + following body text
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
                        c["chunk_index"] = idx * 100 + c["chunk_index"]
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
```

- [ ] **Step 4: Update _extract_pages_docx to return sections**

In `src/chunk.py`, replace `_extract_pages_docx` with:

```python
def _extract_pages_docx(path: Path, course: str) -> list[dict]:
    """Extract text from .docx files, returning one page per heading section.

    Paragraphs with Heading1 or Heading2 styles mark section boundaries.
    If no headings are found, returns a single page with all text.
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
```

- [ ] **Step 5: Run tests to confirm they pass**

Run: `uv run pytest src/tests/test_chunk.py::test_chunk_structured_splits_md_on_headings src/tests/test_chunk.py::test_chunk_structured_no_headings_returns_window_chunks src/tests/test_chunk.py::test_chunk_structured_sets_slide_title_from_heading -v`
Expected: All 3 PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 7: Commit**

```
git add src/chunk.py src/tests/test_chunk.py
git commit -m "feat: chunk_structured for MD/DOCX/outlined PDF, section-aware DOCX extraction"
```

---

## Task 6: Salvage pass

**Files:**
- Modify: `src/router.py`
- Modify: `src/tests/test_router.py`

- [ ] **Step 1: Write failing tests**

Add to `src/tests/test_router.py`:

```python
from src.router import salvage_pass, ChunkingProfile


def _garbled_profile(ratio: float = 0.5) -> ChunkingProfile:
    return ChunkingProfile(
        strategy="slides", avg_tokens_per_page=50.0,
        has_structure=False, garbled_page_ratio=ratio,
        file_extension=".pdf", page_count=2,
    )


def test_salvage_pass_skips_when_no_garbled(tmp_path):
    pages = [{"course": "IS 6410", "filename": "f.pdf", "page": 1,
              "slide_title": "", "text": "word " * 50}]
    profile = _garbled_profile(ratio=0.0)
    stubs = salvage_pass(pages, profile)
    assert stubs == []


def test_salvage_pass_creates_stub_from_title():
    good_title = "Entity Relationship Diagrams"
    garbled_body = "E 8 c I M O xa S e U n fl o ij ri O CM COrAPLICATED OOES WHAT more single chars here"
    page = {
        "course": "IS 6410", "filename": "slides.pdf", "page": 5,
        "slide_title": "ERDs",
        "text": f"{good_title}\n{garbled_body}",
    }
    profile = _garbled_profile(ratio=1.0)
    stubs = salvage_pass([page], profile)
    assert len(stubs) == 1
    assert stubs[0]["is_stub"] is True
    assert stubs[0]["strategy"] == "salvage"
    assert stubs[0]["page"] == 5
    assert "Entity Relationship Diagrams" in stubs[0]["text"]


def test_salvage_pass_skips_quality_pages():
    good_page = {
        "course": "IS 6410", "filename": "slides.pdf", "page": 1,
        "slide_title": "", "text": "word " * 50,
    }
    profile = _garbled_profile(ratio=0.5)
    stubs = salvage_pass([good_page], profile)
    assert stubs == []


def test_salvage_pass_rejects_garbled_title():
    fully_garbled = {
        "course": "IS 6410", "filename": "slides.pdf", "page": 3,
        "slide_title": "",
        "text": "E 8 c I xa S e U n fl o ij ri O CM COrAPLICATED OOES WHAT more single",
    }
    profile = _garbled_profile(ratio=1.0)
    stubs = salvage_pass([fully_garbled], profile)
    assert stubs == []
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest src/tests/test_router.py::test_salvage_pass_creates_stub_from_title -v`
Expected: FAIL — `ImportError: cannot import name 'salvage_pass'`

- [ ] **Step 3: Implement salvage_pass in router.py**

Add to `src/router.py` after `build_profile`:

```python
from src.config import MIN_STUB_TOKENS


def _extract_stub(page: dict) -> dict | None:
    """Try to extract a meaningful title from a garbled page.

    Takes the first non-empty line. Rejects it if it has fewer than
    MIN_STUB_TOKENS meaningful words or too much single-character noise.
    """
    lines = [l.strip() for l in page["text"].splitlines() if l.strip()]
    if not lines:
        return None
    title = lines[0]
    # Strip markdown heading markers if present
    title = re.sub(r'^#{1,3}\s+', '', title).strip()
    if not title:
        return None

    words = [w for w in title.split() if re.sub(r"[^a-zA-Z]", "", w)]
    if len(words) < MIN_STUB_TOKENS:
        return None
    single_char = sum(1 for w in words if len(re.sub(r"[^a-zA-Z]", "", w)) <= 1)
    if words and single_char / len(words) > 0.2:
        return None

    chunk_id = f"{page['course']}_{page['filename']}_p{page['page']}_stub"
    return {
        "id": chunk_id,
        "course": page["course"],
        "filename": page["filename"],
        "page": page["page"],
        "slide_title": page.get("slide_title", ""),
        "chunk_index": 0,
        "text": title,
        "strategy": "salvage",
        "is_stub": True,
    }


def salvage_pass(pages: list[dict], profile: ChunkingProfile) -> list[dict]:
    """Create stub chunks from garbled pages that have extractable titles.

    Only runs when profile.garbled_page_ratio > 0. For each page that fails
    the quality check, attempts to extract the first line as a stub chunk.
    Stub chunks are flagged with is_stub=True and strategy="salvage".
    """
    if profile.garbled_page_ratio == 0.0:
        return []

    stubs = []
    for page in pages:
        if is_quality_text(page["text"]):
            continue  # page was fine, already chunked by main strategy
        stub = _extract_stub(page)
        if stub:
            stubs.append(stub)
    return stubs
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `uv run pytest src/tests/test_router.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```
git add src/router.py src/tests/test_router.py
git commit -m "feat: salvage_pass creates stub chunks from garbled page titles"
```

---

## Task 7: route_and_chunk + refactor build_chunks_from_file

**Files:**
- Modify: `src/router.py`
- Modify: `src/chunk.py`
- Modify: `src/tests/test_router.py`
- Modify: `src/tests/test_chunk.py`

- [ ] **Step 1: Write failing tests for route_and_chunk**

Add to `src/tests/test_router.py`:

```python
from unittest.mock import patch
from src.router import route_and_chunk


def mock_embed(texts):
    return [[0.1] * 768 for _ in texts]


def test_route_and_chunk_slides_strategy():
    profile = ChunkingProfile(
        strategy="slides", avg_tokens_per_page=40.0, has_structure=False,
        garbled_page_ratio=0.0, file_extension=".pdf", page_count=8,
    )
    pages = [_make_pages(1, 30)[0] for _ in range(8)]
    for i, p in enumerate(pages):
        p["page"] = i + 1
    chunks = route_and_chunk(profile, pages, embed_fn=mock_embed)
    assert len(chunks) >= 1
    assert all(c["strategy"] == "slides" for c in chunks)
    assert all(c["is_stub"] is False for c in chunks)


def test_route_and_chunk_semantic_strategy():
    profile = ChunkingProfile(
        strategy="semantic", avg_tokens_per_page=300.0, has_structure=False,
        garbled_page_ratio=0.0, file_extension=".pdf", page_count=2,
    )
    pages = _make_pages(2, 300)
    chunks = route_and_chunk(profile, pages, embed_fn=mock_embed)
    assert len(chunks) >= 1
    assert all(c["strategy"] == "semantic" for c in chunks)


def test_route_and_chunk_structured_strategy():
    profile = ChunkingProfile(
        strategy="structured", avg_tokens_per_page=200.0, has_structure=True,
        garbled_page_ratio=0.0, file_extension=".md", page_count=1,
    )
    pages = [{"course": "IS 6410", "filename": "notes.md", "page": 1,
              "slide_title": "", "text": "# Topic A\n\nSome content here about topic A.\n\n# Topic B\n\nContent about topic B."}]
    chunks = route_and_chunk(profile, pages, embed_fn=mock_embed)
    assert len(chunks) >= 1
    assert all(c["strategy"] == "structured" for c in chunks)


def test_route_and_chunk_includes_salvage_stubs():
    profile = ChunkingProfile(
        strategy="slides", avg_tokens_per_page=40.0, has_structure=False,
        garbled_page_ratio=0.5, file_extension=".pdf", page_count=2,
    )
    good_page = _make_pages(1, 40)[0]
    garbled_page = {
        "course": "IS 6410", "filename": "test.pdf", "page": 2,
        "slide_title": "Project Portfolio Management",
        "text": "Project Portfolio Management\nE 8 c I xa S e U n fl o ij ri O CM COrAPLICATED OOES WHAT single chars here",
    }
    chunks = route_and_chunk(profile, [good_page, garbled_page], embed_fn=mock_embed)
    stubs = [c for c in chunks if c.get("is_stub")]
    assert len(stubs) == 1
    assert "Project Portfolio Management" in stubs[0]["text"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest src/tests/test_router.py::test_route_and_chunk_slides_strategy -v`
Expected: FAIL — `ImportError: cannot import name 'route_and_chunk'`

- [ ] **Step 3: Implement route_and_chunk in router.py**

Add to `src/router.py`:

```python
def route_and_chunk(profile: ChunkingProfile, pages: list[dict], embed_fn) -> list[dict]:
    """Route pages to the appropriate chunking strategy and run salvage pass.

    Returns all chunks (main strategy + salvage stubs). Every chunk has
    'strategy' and 'is_stub' metadata fields.

    Args:
        profile: ChunkingProfile computed by build_profile.
        pages: extracted page dicts from chunk.extract_pages.
        embed_fn: callable(list[str]) -> list[list[float]] for semantic chunking.
    """
    from src.chunk import chunk_slides, chunk_page, chunk_semantic, chunk_structured

    chunks = []

    if profile.strategy == "slides":
        chunks = chunk_slides(pages)
        for c in chunks:
            c.setdefault("strategy", "slides")
            c.setdefault("is_stub", False)

    elif profile.strategy == "structured":
        chunks = chunk_structured(pages)
        # chunk_structured already sets strategy and is_stub

    elif profile.strategy == "semantic":
        for page in pages:
            page_chunks = chunk_semantic(page, embed_fn=embed_fn)
            chunks.extend(page_chunks)

    else:
        # "window" fallback
        for page in pages:
            page_chunks = chunk_page(page)
            for c in page_chunks:
                c.setdefault("strategy", "window")
                c.setdefault("is_stub", False)
            chunks.extend(page_chunks)

    # Salvage pass — adds stub chunks for garbled pages
    stubs = salvage_pass(pages, profile)
    chunks.extend(stubs)

    return chunks
```

- [ ] **Step 4: Refactor build_chunks_from_file in chunk.py**

Replace the current `build_chunks_from_file` function with:

```python
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

    profile = build_profile(path, pages)
    return route_and_chunk(profile, pages, embed_fn=embed_batch)
```

- [ ] **Step 5: Update existing test_chunk.py tests that call build_chunks_from_file**

The existing tests `test_build_chunks_from_file_pdf`, `test_build_chunks_from_file_txt`, `test_build_chunks_from_file_md` check for `id`, `text`, `page` fields. Add `strategy` and `is_stub` checks:

```python
def test_build_chunks_from_file_pdf(sample_pdf):
    chunks = build_chunks_from_file(sample_pdf, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c and "page" in c for c in chunks)
    assert all("strategy" in c and "is_stub" in c for c in chunks)

def test_build_chunks_from_file_txt(tmp_path):
    txt_file = tmp_path / "reading.txt"
    txt_file.write_text(("word " * 600), encoding="utf-8")
    chunks = build_chunks_from_file(txt_file, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c for c in chunks)
    assert all("strategy" in c and "is_stub" in c for c in chunks)

def test_build_chunks_from_file_md(tmp_path):
    md_file = tmp_path / "note.md"
    md_file.write_text("# Topic\n\n" + ("word " * 50), encoding="utf-8")
    chunks = build_chunks_from_file(md_file, course="IS 6410")
    assert len(chunks) >= 1
    assert all("strategy" in c and "is_stub" in c for c in chunks)
```

- [ ] **Step 6: Run all router tests**

Run: `uv run pytest src/tests/test_router.py -v`
Expected: All tests PASS

- [ ] **Step 7: Run full suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 8: Commit**

```
git add src/router.py src/chunk.py src/tests/test_router.py src/tests/test_chunk.py
git commit -m "feat: route_and_chunk integrates all strategies, refactor build_chunks_from_file"
```

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add stub chunk instruction**

Open `CLAUDE.md` and add to the "Citation rules" section:

```markdown
- Chunks with `is_stub: true` in metadata indicate a topic exists in the source document but the content was unreadable (diagram, scanned image). Treat them as evidence of topic presence only — do not make factual claims from stub chunks and do not cite them as factual sources.
```

- [ ] **Step 2: Commit**

```
git add CLAUDE.md
git commit -m "docs: add stub chunk instruction to CLAUDE.md"
```

---

## Task 9: End-to-end verification and push

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest src/tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify ingest pipeline imports cleanly**

Run: `python -c "from src.ingest import run_ingestion; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify router imports cleanly**

Run: `python -c "from src.router import build_profile, route_and_chunk, salvage_pass, ChunkingProfile; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Push to GitHub**

```
git push origin master:main
```

- [ ] **Step 5: After pushing, wipe and re-ingest**

```powershell
Remove-Item -Recurse -Force src\chroma, src\bm25.pkl, src\ingestion_log.json -ErrorAction SilentlyContinue
uv run python src/ingest.py
uv run python src/graph.py
```

Expected: Ingest output now shows strategy per file (semantic/structured/slides) and reports salvaged stubs where applicable.
