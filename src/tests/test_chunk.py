from pathlib import Path
from src.chunk import (
    extract_pages, chunk_page, chunk_slides, is_slide_deck,
    build_chunks_from_file, _extract_pages_text, is_quality_text,
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


def _make_page(course="IS 6410", filename="test.pdf", page=1, text="slide content"):
    return {"course": course, "filename": filename, "page": page,
            "slide_title": "", "text": text}


def test_is_slide_deck_detects_short_pages():
    pages = [_make_page(text="short bullet point") for _ in range(5)]
    assert is_slide_deck(pages) is True


def test_is_slide_deck_rejects_dense_text():
    dense = "word " * 200
    pages = [_make_page(text=dense) for _ in range(3)]
    assert is_slide_deck(pages) is False


def test_chunk_slides_groups_pages():
    pages = [_make_page(page=i, text=f"slide {i} content") for i in range(1, 9)]
    chunks = chunk_slides(pages)
    assert len(chunks) == 8 // SLIDE_PAGES_PER_CHUNK
    assert "slide 1 content" in chunks[0]["text"]
    assert "slide 4 content" in chunks[0]["text"]
    assert "slide 5 content" in chunks[1]["text"]


def test_chunk_slides_id_uses_first_page():
    pages = [_make_page(page=i, text=f"content {i}") for i in range(1, 5)]
    chunks = chunk_slides(pages)
    assert chunks[0]["id"] == "IS 6410_test.pdf_p1_c0"
    assert chunks[0]["page"] == 1


def test_chunk_slides_handles_remainder():
    pages = [_make_page(page=i, text=f"slide {i}") for i in range(1, 7)]
    chunks = chunk_slides(pages)
    assert len(chunks) == 2
    assert "slide 5" in chunks[1]["text"]
    assert "slide 6" in chunks[1]["text"]


def test_build_chunks_from_file_pdf(sample_pdf):
    chunks = build_chunks_from_file(sample_pdf, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c and "page" in c for c in chunks)


def test_build_chunks_from_file_txt(tmp_path):
    txt_file = tmp_path / "reading.txt"
    txt_file.write_text(("word " * 600), encoding="utf-8")
    chunks = build_chunks_from_file(txt_file, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c for c in chunks)


def test_build_chunks_from_file_md(tmp_path):
    md_file = tmp_path / "note.md"
    md_file.write_text("# Topic\n\n" + ("word " * 50), encoding="utf-8")
    chunks = build_chunks_from_file(md_file, course="IS 6410")
    assert len(chunks) >= 1
