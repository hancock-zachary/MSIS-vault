import pytest
from unittest.mock import patch, MagicMock
from src.index import (
    upsert_chunks, query_dense, build_bm25, query_bm25, load_bm25
)


@pytest.fixture
def mock_collection():
    col = MagicMock()
    col.query.return_value = {
        "ids": [["IS 6410_test.pdf_p1_c0"]],
        "documents": [["An ERD models data entities."]],
        "metadatas": [[{"course": "IS 6410", "filename": "test.pdf", "page": 1}]],
        "distances": [[0.1]],
    }
    return col


def test_query_dense_returns_ranked_chunks(mock_collection, sample_chunks):
    results = query_dense(mock_collection, query_vector=[0.1]*768, top_k=5)
    assert len(results) == 1
    assert results[0]["id"] == "IS 6410_test.pdf_p1_c0"
    assert results[0]["score"] >= 0.0


def test_build_and_query_bm25(tmp_path, sample_chunks):
    # Add two unrelated chunks so BM25 IDF produces positive scores
    second_chunk = {
        "id": "IS 6410_test.pdf_p2_c0",
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 2,
        "slide_title": "Supply Chain",
        "chunk_index": 0,
        "text": "Supply chain management involves logistics and operations.",
    }
    third_chunk = {
        "id": "IS 6410_test.pdf_p3_c0",
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 3,
        "slide_title": "Normalization",
        "chunk_index": 0,
        "text": "Normalization reduces redundancy in database design.",
    }
    chunks = sample_chunks + [second_chunk, third_chunk]
    bm25_path = tmp_path / "bm25.pkl"
    index, corpus = build_bm25(chunks, bm25_path)
    results = query_bm25(index, corpus, chunks, "ERD entities relationships", top_k=5)
    assert len(results) >= 1
    assert results[0]["id"] == "IS 6410_test.pdf_p1_c0"


def test_load_bm25_roundtrip(tmp_path, sample_chunks):
    # Add two unrelated chunks so BM25 IDF produces positive scores
    second_chunk = {
        "id": "IS 6410_test.pdf_p2_c0",
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 2,
        "slide_title": "Supply Chain",
        "chunk_index": 0,
        "text": "Supply chain management involves logistics and operations.",
    }
    third_chunk = {
        "id": "IS 6410_test.pdf_p3_c0",
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 3,
        "slide_title": "Normalization",
        "chunk_index": 0,
        "text": "Normalization reduces redundancy in database design.",
    }
    chunks = sample_chunks + [second_chunk, third_chunk]
    bm25_path = tmp_path / "bm25.pkl"
    build_bm25(chunks, bm25_path)
    loaded_index, loaded_corpus, loaded_chunks = load_bm25(bm25_path)
    results = query_bm25(loaded_index, loaded_corpus, loaded_chunks, "entities", top_k=5)
    assert len(results) >= 1


def test_upsert_chunks_calls_collection_upsert(mock_collection, sample_chunks):
    embeddings = [[0.1] * 768]
    upsert_chunks(mock_collection, sample_chunks, embeddings)
    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args[1]
    assert call_kwargs["ids"] == ["IS 6410_test.pdf_p1_c0"]
    assert call_kwargs["documents"] == ["An ERD models data entities and their relationships."]
    # metadata should not contain 'id' or 'text' keys
    assert "id" not in call_kwargs["metadatas"][0]
    assert "text" not in call_kwargs["metadatas"][0]
