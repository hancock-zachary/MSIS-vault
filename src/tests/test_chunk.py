from pathlib import Path
from src.chunk import (
    extract_pages, chunk_page, chunk_slides, is_slide_deck,
    build_chunks_from_file, _extract_pages_text, is_quality_text,
    enrich_for_embedding,
)
from src.config import SLIDE_PAGES_PER_CHUNK


def test_is_quality_text_accepts_normal_text():
    text = "The five Scrum events are Sprint Planning, Daily Scrum, Sprint Review, Sprint Retrospective, and the Sprint itself."
    assert is_quality_text(text) is True

def test_is_quality_text_rejects_garbled():
    garbled = "E 8 c I M O O 75 2 1 S xa S e U n E fl o ij ri O CM o COrAPLICATED THAT NO ONE KNOWS WHO OOES WHAT"
    assert is_quality_text(garbled) is False

def test_is_quality_text_rejects_too_short():
    assert is_quality_text("too short") is False


def test_extract_pages_pdf(sample_pdf):
    pages = extract_pages(sample_pdf, course="IS 6410")
    assert len(pages) >= 1
    assert pages[0]["page"] == 1
    assert "Entity-Relationship" in pages[0]["text"]
    assert pages[0]["course"] == "IS 6410"
    assert pages[0]["filename"] == sample_pdf.name


def test_extract_pages_pdf_outline_titles_are_strings_on_correct_pages(tmp_path):
    # get_toc() entries are [level, title, 1-based page]. The extractor must
    # attach the title string (not the int level) to the right page.
    import fitz
    pdf_path = tmp_path / "outlined.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Dense content for page {i + 1} about database design.")
    doc.set_toc([[1, "Introduction", 1], [1, "Methods", 3]])
    doc.save(str(pdf_path))
    doc.close()

    pages = extract_pages(pdf_path, course="IS 6410")
    assert pages[0]["slide_title"] == "Introduction"
    assert pages[1]["slide_title"] == ""
    assert pages[2]["slide_title"] == "Methods"


def test_extract_pages_txt(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("This is a plain text note about project management.", encoding="utf-8")
    pages = extract_pages(txt_file, course="IS 6410")
    assert len(pages) == 1
    assert "project management" in pages[0]["text"]
    assert pages[0]["page"] == 1


def test_extract_pages_md(tmp_path):
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Sprint Planning\n\nSprint planning is the first Scrum event.", encoding="utf-8")
    pages = extract_pages(md_file, course="IS 6410")
    assert len(pages) == 1
    assert "Sprint planning" in pages[0]["text"]


def test_extract_pages_md_strips_frontmatter(tmp_path):
    md_file = tmp_path / "note.md"
    md_file.write_text("---\ntags: [IS-6410]\ndate: 2026-06-01\n---\n\nActual content here.", encoding="utf-8")
    pages = extract_pages(md_file, course="IS 6410")
    assert "Actual content here" in pages[0]["text"]
    assert "tags:" not in pages[0]["text"]


def test_extract_pages_unsupported_raises(tmp_path):
    bad_file = tmp_path / "file.pptx"
    bad_file.touch()
    try:
        extract_pages(bad_file, course="IS 6410")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported" in str(e)


def test_extract_pages_doc_raises_helpful_error(tmp_path):
    doc_file = tmp_path / "old.doc"
    doc_file.touch()
    try:
        extract_pages(doc_file, course="IS 6410")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert ".docx" in str(e)


def test_chunk_page_single_chunk_for_short_text():
    page = {
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 1,
        "slide_title": "ERDs",
        "text": "Short text that fits in one chunk.",
    }
    chunks = chunk_page(page)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "IS 6410_test.pdf_p1_c0"
    assert chunks[0]["text"] == "Short text that fits in one chunk."


def test_chunk_page_overlaps_long_text():
    long_text = "word " * 600
    page = {
        "course": "IS 6410", "filename": "test.pdf",
        "page": 2, "slide_title": "Long Slide", "text": long_text,
    }
    chunks = chunk_page(page)
    assert len(chunks) >= 2
    words_0 = chunks[0]["text"].split()
    words_1 = chunks[1]["text"].split()
    assert words_0[-1] == words_1[0] or words_0[-10:] == words_1[:10]


def _make_page(course="IS 6410", filename="test.pdf", page=1, text="slide content",
               slide_title=""):
    return {"course": course, "filename": filename, "page": page,
            "slide_title": slide_title, "text": text}


def test_is_slide_deck_detects_short_pages():
    pages = [_make_page(text="short bullet point") for _ in range(5)]
    assert is_slide_deck(pages) is True


def test_is_slide_deck_rejects_dense_text():
    dense = "word " * 200
    pages = [_make_page(text=dense) for _ in range(3)]
    assert is_slide_deck(pages) is False


def test_chunk_slides_overlaps_consecutive_groups():
    # 8 pages, group size 4, overlap 1 → groups 1-4, 4-7, 7-8. The boundary
    # page appears in both neighbouring chunks so related slides aren't cut.
    pages = [_make_page(page=i, text=f"slide {i} content") for i in range(1, 9)]
    chunks = chunk_slides(pages)
    assert len(chunks) == 3
    assert "slide 4 content" in chunks[0]["text"]
    assert "slide 4 content" in chunks[1]["text"]
    assert (chunks[0]["page_start"], chunks[0]["page_end"]) == (1, 4)
    assert (chunks[1]["page_start"], chunks[1]["page_end"]) == (4, 7)
    assert (chunks[2]["page_start"], chunks[2]["page_end"]) == (7, 8)


def test_chunk_slides_id_uses_first_page():
    pages = [_make_page(page=i, text=f"content {i}") for i in range(1, 5)]
    chunks = chunk_slides(pages)
    assert chunks[0]["id"] == "IS 6410_test.pdf_p1_c0"
    assert chunks[0]["page"] == 1


def test_chunk_slides_single_group_when_pages_fit():
    # Exactly one group — no degenerate overlap-only trailing chunk.
    pages = [_make_page(page=i, text=f"slide {i}") for i in range(1, 5)]
    chunks = chunk_slides(pages)
    assert len(chunks) == 1


def test_chunk_slides_handles_remainder():
    pages = [_make_page(page=i, text=f"slide {i}") for i in range(1, 7)]
    chunks = chunk_slides(pages)
    assert len(chunks) == 2
    assert "slide 5" in chunks[1]["text"]
    assert "slide 6" in chunks[1]["text"]


def test_chunk_slides_splits_at_outline_sections():
    # Titled pages start new sections; chunks never span a section boundary
    # even when the fixed group size would.
    pages = [
        _make_page(page=1, text="intro a", slide_title="Introduction"),
        _make_page(page=2, text="intro b"),
        _make_page(page=3, text="intro c"),
        _make_page(page=4, text="methods a", slide_title="Methods"),
        _make_page(page=5, text="methods b"),
        _make_page(page=6, text="methods c"),
    ]
    chunks = chunk_slides(pages)
    assert len(chunks) == 2
    assert (chunks[0]["page_start"], chunks[0]["page_end"]) == (1, 3)
    assert (chunks[1]["page_start"], chunks[1]["page_end"]) == (4, 6)
    assert chunks[0]["slide_title"] == "Introduction"
    assert chunks[1]["slide_title"] == "Methods"
    assert "methods" not in chunks[0]["text"]


def test_chunk_slides_windows_large_sections_with_title():
    # A long section still gets windowed, and every window carries the
    # section title (feeds the embedding context header).
    pages = [_make_page(page=1, text="intro 1", slide_title="Introduction")] + [
        _make_page(page=i, text=f"intro {i}") for i in range(2, 10)
    ]
    chunks = chunk_slides(pages)
    assert len(chunks) == 3
    assert (chunks[0]["page_start"], chunks[0]["page_end"]) == (1, 4)
    assert (chunks[1]["page_start"], chunks[1]["page_end"]) == (4, 7)
    assert (chunks[2]["page_start"], chunks[2]["page_end"]) == (7, 9)
    assert all(c["slide_title"] == "Introduction" for c in chunks)


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


# ---------------------------------------------------------------------------
# chunk_semantic tests
# ---------------------------------------------------------------------------
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
    assert all("strategy" in c for c in chunks)

def test_chunk_semantic_splits_on_low_similarity():
    text = (
        "Scrum is an agile framework for managing software development. "
        "Supply chain management involves logistics and procurement."
    )
    page = _make_dense_page(text)

    call_count = [0]
    def mock_embed(texts):
        call_count[0] += len(texts)
        vectors = []
        for i, t in enumerate(texts):
            v = [0.0] * 768
            v[i % 768] = 1.0
            vectors.append(v)
        return vectors

    chunks = chunk_semantic(page, embed_fn=mock_embed)
    assert call_count[0] > 0
    assert all(c["is_stub"] is False for c in chunks)


# ---------------------------------------------------------------------------
# chunk_structured tests
# ---------------------------------------------------------------------------
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


def test_enrich_for_embedding_prepends_context_header():
    chunk = {
        "course": "IS 6410",
        "filename": "week3-erd.pdf",
        "slide_title": "ER Diagrams",
        "text": "An ERD models entities.",
    }
    enriched = enrich_for_embedding(chunk)
    assert enriched.startswith("IS 6410 - week3-erd - ER Diagrams")
    assert enriched.endswith("An ERD models entities.")


def test_enrich_for_embedding_omits_empty_slide_title():
    chunk = {
        "course": "IS 6410",
        "filename": "week3-erd.pdf",
        "slide_title": "",
        "text": "An ERD models entities.",
    }
    enriched = enrich_for_embedding(chunk)
    assert enriched.startswith("IS 6410 - week3-erd\n")
