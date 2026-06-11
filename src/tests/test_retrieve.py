import pytest
from unittest.mock import MagicMock, patch
from src.config import STUB_RRF_MULTIPLIER
from src.retrieve import (
    apply_stub_penalty, expand_neighbors, hybrid_retrieve,
    reciprocal_rank_fusion, deduplicate_by_id, select_neighbors,
)


def _nchunk(cid, page, text="text", filename="deck.pdf", chunk_index=0,
            page_end=None, is_stub=False):
    return {
        "id": cid, "filename": filename, "page": page,
        "page_start": page, "page_end": page_end or page,
        "chunk_index": chunk_index, "text": text, "is_stub": is_stub,
    }


# ---------------------------------------------------------------------------
# select_neighbors
# ---------------------------------------------------------------------------

def test_select_neighbors_picks_page_adjacent_chunks():
    winner = _nchunk("w", page=3)
    same_file = [
        _nchunk("far_before", page=1),
        _nchunk("before", page=2),
        winner,
        _nchunk("after", page=4),
        _nchunk("far_after", page=6),
    ]
    ids = [n["id"] for n in select_neighbors(winner, same_file)]
    assert ids == ["before", "after"]


def test_select_neighbors_includes_same_page_siblings_in_index_order():
    winner = _nchunk("w", page=3, chunk_index=1)
    same_file = [
        _nchunk("sib2", page=3, chunk_index=2),
        _nchunk("sib0", page=3, chunk_index=0),
        winner,
    ]
    ids = [n["id"] for n in select_neighbors(winner, same_file)]
    assert ids == ["sib0", "sib2"]


def test_select_neighbors_excludes_stubs():
    winner = _nchunk("w", page=3)
    same_file = [winner, _nchunk("stub", page=2, is_stub=True)]
    assert select_neighbors(winner, same_file) == []


def test_select_neighbors_respects_page_ranges():
    # Slide chunk spanning pages 5-8: ranges touching [4, 9] are neighbors.
    winner = _nchunk("w", page=5, page_end=8)
    same_file = [
        winner,
        _nchunk("prev_group", page=1, page_end=4),
        _nchunk("next_group", page=9, page_end=12),
        _nchunk("distant", page=14, page_end=16),
    ]
    ids = [n["id"] for n in select_neighbors(winner, same_file)]
    assert ids == ["prev_group", "next_group"]


# ---------------------------------------------------------------------------
# expand_neighbors
# ---------------------------------------------------------------------------

def _mock_collection_with(chunks):
    col = MagicMock()
    col.get.return_value = {
        "ids": [c["id"] for c in chunks],
        "documents": [c["text"] for c in chunks],
        "metadatas": [
            {k: v for k, v in c.items() if k not in ("id", "text")} for c in chunks
        ],
    }
    return col


def test_expand_neighbors_merges_text_in_page_order():
    file_chunks = [
        _nchunk("p1", page=1, text="page one"),
        _nchunk("p2", page=2, text="page two"),
        _nchunk("p3", page=3, text="page three"),
    ]
    winner = dict(file_chunks[1])
    with patch("src.retrieve.get_collection",
               return_value=_mock_collection_with(file_chunks)):
        expanded = expand_neighbors([winner])
    assert expanded[0]["text"] == "page one\n\npage two\n\npage three"
    assert expanded[0]["page_start"] == 1
    assert expanded[0]["page_end"] == 3
    assert expanded[0]["id"] == "p2"


def test_expand_neighbors_claims_each_neighbor_once():
    file_chunks = [
        _nchunk("p1", page=1, text="page one"),
        _nchunk("p2", page=2, text="page two"),
        _nchunk("p3", page=3, text="page three"),
    ]
    winners = [dict(file_chunks[0]), dict(file_chunks[1])]
    with patch("src.retrieve.get_collection",
               return_value=_mock_collection_with(file_chunks)):
        expanded = expand_neighbors(winners)
    # Winners never absorb each other, and p3 is claimed by p2 only.
    assert expanded[0]["text"] == "page one"
    assert expanded[1]["text"] == "page two\n\npage three"


def test_hybrid_retrieve_passes_course_filter_to_dense():
    with patch("src.retrieve.get_collection"), \
         patch("src.retrieve.embed_text", return_value=[0.1] * 768), \
         patch("src.retrieve.load_bm25", side_effect=FileNotFoundError), \
         patch("src.retrieve.query_dense", return_value=[]) as mock_dense:
        hybrid_retrieve(["what is an ERD?"], course="IS 6410")
    assert mock_dense.call_args.kwargs["where"] == {"course": "IS 6410"}


def test_hybrid_retrieve_no_filter_by_default():
    with patch("src.retrieve.get_collection"), \
         patch("src.retrieve.embed_text", return_value=[0.1] * 768), \
         patch("src.retrieve.load_bm25", side_effect=FileNotFoundError), \
         patch("src.retrieve.query_dense", return_value=[]) as mock_dense:
        hybrid_retrieve(["what is an ERD?"])
    assert mock_dense.call_args.kwargs["where"] is None


def test_apply_stub_penalty_demotes_stubs():
    chunks = [
        {"id": "stub", "is_stub": True, "rrf_score": 0.05},
        {"id": "real", "is_stub": False, "rrf_score": 0.04},
    ]
    result = apply_stub_penalty(chunks)
    assert [c["id"] for c in result] == ["real", "stub"]
    assert result[1]["rrf_score"] == pytest.approx(0.05 * STUB_RRF_MULTIPLIER)


def test_apply_stub_penalty_leaves_non_stubs_untouched():
    chunks = [
        {"id": "a", "is_stub": False, "rrf_score": 0.05},
        {"id": "b", "rrf_score": 0.04},  # missing is_stub treated as non-stub
    ]
    result = apply_stub_penalty(chunks)
    assert [c["id"] for c in result] == ["a", "b"]
    assert result[0]["rrf_score"] == pytest.approx(0.05)
    assert result[1]["rrf_score"] == pytest.approx(0.04)


def test_rrf_higher_ranked_gets_higher_score():
    list_a = [{"id": "a", "text": "t"}, {"id": "b", "text": "t"}]
    list_b = [{"id": "a", "text": "t"}, {"id": "c", "text": "t"}]
    fused = reciprocal_rank_fusion([list_a, list_b])
    ids = [c["id"] for c in fused]
    # "a" appears rank 1 in both lists so should be first
    assert ids[0] == "a"


def test_rrf_merges_all_unique_chunks():
    list_a = [{"id": "a", "text": "t"}, {"id": "b", "text": "t"}]
    list_b = [{"id": "c", "text": "t"}]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert {c["id"] for c in fused} == {"a", "b", "c"}


def test_deduplicate_keeps_first_occurrence():
    chunks = [
        {"id": "a", "text": "first"},
        {"id": "a", "text": "duplicate"},
        {"id": "b", "text": "unique"},
    ]
    result = deduplicate_by_id(chunks)
    assert len(result) == 2
    assert result[0]["text"] == "first"
