import pytest
from pathlib import Path
from unittest.mock import patch
from src.router import ChunkingProfile, build_profile, salvage_pass, route_and_chunk


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


# ---------------------------------------------------------------------------
# route_and_chunk tests
# ---------------------------------------------------------------------------

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
