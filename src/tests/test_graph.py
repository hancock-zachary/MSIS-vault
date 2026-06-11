from unittest.mock import MagicMock

from src.graph import _get_excerpts, _most_central_indices, _write_page


def test_most_central_indices_picks_chunks_near_centroid():
    embeddings = [
        [0.0, 1.0],    # outlier (e.g. title page)
        [1.0, 0.0],    # cluster
        [0.95, 0.05],  # cluster
    ]
    picked = _most_central_indices(embeddings, n=2)
    assert set(picked) == {1, 2}


def test_most_central_indices_returns_all_when_fewer_than_n():
    assert _most_central_indices([[1.0, 0.0]], n=2) == [0]


def test_get_excerpts_returns_central_chunks_in_page_order():
    # Page 1 is an embedding outlier (title page) and the stub sits exactly
    # on the centroid — both must be skipped in favour of the cluster.
    col = MagicMock()
    col.get.return_value = {
        "ids": ["c1", "c2", "c3", "stub"],
        "documents": ["title page", "core content B", "core content A", "stub title"],
        "metadatas": [
            {"page": 1, "is_stub": False},
            {"page": 3, "is_stub": False},
            {"page": 2, "is_stub": False},
            {"page": 4, "is_stub": True},
        ],
        "embeddings": [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.95, 0.05],
            [0.65, 0.35],
        ],
    }
    excerpts = _get_excerpts(col, "deck.pdf", n=2)
    assert excerpts == ["core content A", "core content B"]  # page order



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
