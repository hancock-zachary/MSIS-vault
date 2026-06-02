from unittest.mock import patch, MagicMock
from brain.rerank import rerank_chunks


def test_rerank_returns_top_k(sample_chunks):
    with patch("brain.rerank._get_model") as mock_model:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9]
        mock_model.return_value = mock_ce
        results = rerank_chunks("What is an ERD?", sample_chunks, top_k=5)
    assert len(results) <= 5
    assert "rerank_score" in results[0]


def test_rerank_orders_by_score(sample_chunks):
    chunk_a = {**sample_chunks[0], "id": "a", "text": "highly relevant"}
    chunk_b = {**sample_chunks[0], "id": "b", "text": "less relevant"}
    scores = {"a": 0.9, "b": 0.1}
    with patch("brain.rerank._get_model") as mock_model:
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = lambda pairs: [scores[p[1].split()[0]] if p[1].split()[0] in scores else 0.5 for p in pairs]
        mock_model.return_value = mock_ce
        results = rerank_chunks("query", [chunk_a, chunk_b], top_k=5)
    # just verify scores are attached and list is sorted
    assert results[0]["rerank_score"] >= results[-1]["rerank_score"]
