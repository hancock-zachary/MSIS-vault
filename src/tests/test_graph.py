from src.graph import _write_page


def test_write_page_includes_source_type_in_frontmatter(tmp_path):
    page_path = tmp_path / "Week 3 - ERDs.md"
    _write_page(
        page_path,
        filename="Week 3 - ERDs.pdf",
        course="IS 6410",
        chunk_count=12,
        related=[],
        excerpts=["An ERD models entities."],
        source_type="slides",
    )
    content = page_path.read_text(encoding="utf-8")
    assert "source_type: slides" in content
    assert content.index("source_type:") < content.index("# Week 3 - ERDs")
