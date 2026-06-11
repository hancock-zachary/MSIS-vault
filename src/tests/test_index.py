import pytest
from unittest.mock import patch, MagicMock
from src.index import (
    upsert_chunks, query_dense, build_bm25, query_bm25, load_bm25, tokenize,
    rebuild_bm25_from_collection,
)


def test_rebuild_bm25_from_collection(tmp_path):
    col = MagicMock()
    col.get.return_value = {
        "ids": ["c1", "c2", "c3"],
        "documents": [
            "An ERD models data entities.",
            "Supply chain logistics overview.",
            "Normalization reduces redundancy.",
        ],
        "metadatas": [
            {"filename": "a.pdf", "page": 1},
            {"filename": "b.pdf", "page": 2},
            {"filename": "c.pdf", "page": 3},
        ],
    }
    path = tmp_path / "bm25.pkl"
    count = rebuild_bm25_from_collection(col, path)
    assert count == 3
    index, corpus, chunks = load_bm25(path)
    results = query_bm25(index, corpus, chunks, "normalization", top_k=5)
    assert results[0]["id"] == "c3"
    assert results[0]["filename"] == "c.pdf"


def test_rebuild_bm25_from_empty_collection_removes_stale_pickle(tmp_path):
    col = MagicMock()
    col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    path = tmp_path / "bm25.pkl"
    path.write_bytes(b"stale")
    count = rebuild_bm25_from_collection(col, path)
    assert count == 0
    assert not path.exists()


def test_tokenize_strips_punctuation():
    assert tokenize("Systems, analysis!") == ["systems", "analysis"]


def test_tokenize_removes_stopwords():
    assert tokenize("the analysis of the system") == ["analysis", "system"]


def test_tokenize_keeps_numbers_and_acronyms():
    assert tokenize("ERD 101 covers WBS.") == ["erd", "101", "covers", "wbs"]


def test_tokenize_splits_hyphenated_terms():
    assert tokenize("entity-relationship") == ["entity", "relationship"]


def test_bm25_matches_despite_punctuation(tmp_path, sample_chunks):
    # "entities" appears as "entities." (with punctuation) in other chunks —
    # naive whitespace tokenization would fail to match the bare query term.
    punctuated = {
        "id": "IS 6410_test.pdf_p4_c0",
        "course": "IS 6410", "filename": "test.pdf", "page": 4,
        "slide_title": "", "chunk_index": 0,
        "text": "Cardinality constrains (entities); see: relationships, attributes.",
    }
    filler = {
        "id": "IS 6410_test.pdf_p5_c0",
        "course": "IS 6410", "filename": "test.pdf", "page": 5,
        "slide_title": "", "chunk_index": 0,
        "text": "Supply chain logistics and operations management overview.",
    }
    filler2 = {
        "id": "IS 6410_test.pdf_p6_c0",
        "course": "IS 6410", "filename": "test.pdf", "page": 6,
        "slide_title": "", "chunk_index": 0,
        "text": "Normalization reduces redundancy in database design.",
    }
    chunks = [punctuated, filler, filler2]
    index, corpus = build_bm25(chunks, tmp_path / "bm25.pkl")
    results = query_bm25(index, corpus, chunks, "cardinality", top_k=5)
    assert len(results) >= 1
    assert results[0]["id"] == "IS 6410_test.pdf_p4_c0"


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


def test_query_dense_passes_course_filter(mock_collection):
    query_dense(mock_collection, query_vector=[0.1] * 768, top_k=5,
                where={"course": "IS 6410"})
    assert mock_collection.query.call_args.kwargs["where"] == {"course": "IS 6410"}


def _course_chunk(cid, course, text):
    return {"id": cid, "course": course, "filename": f"{cid}.pdf", "page": 1,
            "slide_title": "", "chunk_index": 0, "text": text}


def test_query_bm25_filters_by_course(tmp_path):
    chunks = [
        _course_chunk("c1", "IS 6410", "Cardinality constrains entity relationships."),
        _course_chunk("c2", "OSC 6660", "Cardinality also appears in operations material."),
        _course_chunk("c3", "IS 6410", "Normalization reduces redundancy in databases."),
        _course_chunk("c4", "OSC 6660", "Supply chain logistics and operations overview."),
        _course_chunk("c5", "IS 6410", "Agile sprints organize development work."),
    ]
    index, corpus = build_bm25(chunks, tmp_path / "bm25.pkl")
    results = query_bm25(index, corpus, chunks, "cardinality", top_k=5, course="IS 6410")
    assert len(results) >= 1
    assert all(r["course"] == "IS 6410" for r in results)
    assert results[0]["id"] == "c1"


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
