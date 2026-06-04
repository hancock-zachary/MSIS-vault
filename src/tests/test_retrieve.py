from src.retrieve import reciprocal_rank_fusion, deduplicate_by_id


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
