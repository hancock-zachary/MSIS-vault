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
    bad_text = "E 8 c I M O xa S e U n fl o ij ri O CM COrAPLICATED OOES WHAT single chars here"
    pages = [
        {"course": "IS 6410", "filename": "mixed.pdf", "page": 1,
         "slide_title": "", "text": good_text},
        {"course": "IS 6410", "filename": "mixed.pdf", "page": 2,
         "slide_title": "", "text": bad_text},
    ]
    profile = build_profile(pdf, pages)
    assert profile.garbled_page_ratio == pytest.approx(0.5)
