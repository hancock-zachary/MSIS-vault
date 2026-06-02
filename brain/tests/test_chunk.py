from brain.chunk import extract_pages, chunk_page, chunk_slides, is_slide_deck, build_chunks_from_pdf
from brain.config import SLIDE_PAGES_PER_CHUNK


def test_extract_pages_returns_page_dicts(sample_pdf):
    pages = extract_pages(sample_pdf, course="IS 6410")
    assert len(pages) >= 1
    assert pages[0]["page"] == 1
    assert "Entity-Relationship" in pages[0]["text"]
    assert pages[0]["course"] == "IS 6410"
    assert pages[0]["filename"] == sample_pdf.name


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
    long_text = "word " * 600  # ~600 tokens
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
    dense = "word " * 200  # ~200 tokens per page — above threshold
    pages = [_make_page(text=dense) for _ in range(3)]
    assert is_slide_deck(pages) is False


def test_chunk_slides_groups_pages():
    pages = [_make_page(page=i, text=f"slide {i} content") for i in range(1, 9)]
    chunks = chunk_slides(pages)
    # 8 pages / 4 per chunk = 2 chunks
    assert len(chunks) == 8 // SLIDE_PAGES_PER_CHUNK
    # each chunk contains text from all its pages
    assert "slide 1 content" in chunks[0]["text"]
    assert "slide 4 content" in chunks[0]["text"]
    assert "slide 5 content" in chunks[1]["text"]


def test_chunk_slides_id_uses_first_page():
    pages = [_make_page(page=i, text=f"content {i}") for i in range(1, 5)]
    chunks = chunk_slides(pages)
    assert chunks[0]["id"] == "IS 6410_test.pdf_p1_c0"
    assert chunks[0]["page"] == 1


def test_chunk_slides_handles_remainder():
    # 6 pages with SLIDE_PAGES_PER_CHUNK=4 → 2 chunks (4 + 2)
    pages = [_make_page(page=i, text=f"slide {i}") for i in range(1, 7)]
    chunks = chunk_slides(pages)
    assert len(chunks) == 2
    assert "slide 5" in chunks[1]["text"]
    assert "slide 6" in chunks[1]["text"]


def test_build_chunks_from_pdf_detects_slides(sample_pdf):
    # sample_pdf has very short text — should be detected as slide deck
    chunks = build_chunks_from_pdf(sample_pdf, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c and "page" in c for c in chunks)
