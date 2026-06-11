from unittest.mock import patch, MagicMock
import pytest
from src.rerank import rerank_chunks


def test_rerank_returns_top_k(sample_chunks):
    with patch("src.rerank._get_model") as mock_model:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9]
        mock_model.return_value = mock_ce
        results = rerank_chunks("What is an ERD?", sample_chunks, top_k=5)
    assert len(results) <= 5
    assert "rerank_score" in results[0]


def test_rerank_orders_by_score(sample_chunks):
    chunk_a = {**sample_chunks[0], "id": "a", "text": "highly relevant"}
    chunk_b = {**sample_chunks[0], "id": "b", "text": "less relevant"}
    with patch("src.rerank._get_model") as mock_model:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.1]
        mock_model.return_value = mock_ce
        results = rerank_chunks("query", [chunk_a, chunk_b], top_k=5)
    assert results[0]["rerank_score"] == pytest.approx(0.9)
    assert results[1]["rerank_score"] == pytest.approx(0.1)
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]


def test_rerank_places_stubs_after_content_chunks(sample_chunks):
    # Stubs are salvaged titles — keyword-dense, so the cross-encoder can
    # score them highly. They must never outrank real content chunks.
    stub = {**sample_chunks[0], "id": "stub", "text": "Entity Relationship Diagrams", "is_stub": True}
    real = {**sample_chunks[0], "id": "real", "text": "An ERD models data entities.", "is_stub": False}
    with patch("src.rerank._get_model") as mock_model:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [9.0, 2.0]  # stub scores higher
        mock_model.return_value = mock_ce
        results = rerank_chunks("what is an ERD?", [stub, real], top_k=2)
    assert results[0]["id"] == "real"
    assert results[1]["id"] == "stub"


def test_rerank_does_not_mutate_input(sample_chunks):
    with patch("src.rerank._get_model") as mock_model:
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9]
        mock_model.return_value = mock_ce
        original_keys = set(sample_chunks[0].keys())
        rerank_chunks("query", sample_chunks, top_k=5)
    assert set(sample_chunks[0].keys()) == original_keys
