from src.query import detect_course, format_context

_COURSES = ["IS 6410", "OSC 6660"]


def test_detect_course_single_mention():
    assert detect_course("In IS 6410, what is an ERD?", _COURSES) == "IS 6410"


def test_detect_course_is_case_insensitive():
    assert detect_course("what does is 6410 say about ERDs?", _COURSES) == "IS 6410"


def test_detect_course_none_when_absent():
    assert detect_course("what is an ERD?", _COURSES) is None


def test_detect_course_none_when_multiple_mentioned():
    # Cross-course questions must not be scoped to either course.
    q = "compare IS 6410 systems analysis with OSC 6660 supply chains"
    assert detect_course(q, _COURSES) is None


def test_format_context_shows_single_page():
    chunks = [{"filename": "week3.pdf", "page": 5, "course": "IS 6410", "text": "ERD content"}]
    output = format_context("what is an ERD?", chunks)
    assert "week3.pdf, page 5" in output
    assert "ERD content" in output


def test_format_context_shows_source_type():
    chunks = [{"filename": "week3.pdf", "page": 5, "course": "IS 6410",
               "source_type": "slides", "text": "ERD content"}]
    output = format_context("question", chunks)
    assert "course: IS 6410 · slides" in output


def test_format_context_hides_unknown_source_type():
    chunks = [{"filename": "week3.pdf", "page": 5, "course": "IS 6410",
               "source_type": "unknown", "text": "ERD content"}]
    output = format_context("question", chunks)
    assert "unknown" not in output


def test_format_context_shows_page_range_for_slide_chunks():
    chunks = [{
        "filename": "deck.pdf", "page": 1, "page_start": 1, "page_end": 4,
        "course": "IS 6410", "text": "slide content",
    }]
    output = format_context("question", chunks)
    assert "deck.pdf, pages 1-4" in output
