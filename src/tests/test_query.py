from src.query import format_context


def test_format_context_shows_single_page():
    chunks = [{"filename": "week3.pdf", "page": 5, "course": "IS 6410", "text": "ERD content"}]
    output = format_context("what is an ERD?", chunks)
    assert "week3.pdf, page 5" in output
    assert "ERD content" in output


def test_format_context_shows_page_range_for_slide_chunks():
    chunks = [{
        "filename": "deck.pdf", "page": 1, "page_start": 1, "page_end": 4,
        "course": "IS 6410", "text": "slide content",
    }]
    output = format_context("question", chunks)
    assert "deck.pdf, pages 1-4" in output
