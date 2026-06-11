import pytest
from unittest.mock import patch
from src.config import STUB_RRF_MULTIPLIER
from src.retrieve import (
    apply_stub_penalty, hybrid_retrieve, reciprocal_rank_fusion, deduplicate_by_id,
)


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
