from brain.chunk import extract_pages, chunk_page, build_chunks_from_pdf


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
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 2,
        "slide_title": "Long Slide",
        "text": long_text,
    }
    chunks = chunk_page(page)
    assert len(chunks) >= 2
    # overlap: last tokens of chunk 0 appear at start of chunk 1
    words_0 = chunks[0]["text"].split()
    words_1 = chunks[1]["text"].split()
    assert words_0[-1] == words_1[0] or words_0[-10:] == words_1[:10]


def test_build_chunks_from_pdf(sample_pdf):
    chunks = build_chunks_from_pdf(sample_pdf, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c and "page" in c for c in chunks)
