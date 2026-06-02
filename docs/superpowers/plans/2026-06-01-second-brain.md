# Second Brain — Professional RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a professional hybrid RAG pipeline over Obsidian vault PDFs with agentic query rewriting, RRF fusion, cross-encoder reranking, and citation verification — queryable via Claude Code CLI.

**Architecture:** PDFs are extracted and dual-indexed into ChromaDB (dense) and BM25 (sparse). At query time, Claude rewrites the question into multiple variants, hybrid retrieval runs per variant and fuses results via RRF, a cross-encoder reranks candidates, and a citation verifier checks every factual claim in Claude's response before it reaches the user.

**Tech Stack:** Python 3.11+, pymupdf, tiktoken, chromadb, rank_bm25, sentence-transformers, requests (Ollama), pytest

---

## File Structure

```
Vault/University of Utah - MSIS/
  brain/
    config.py          — all paths, constants, tunable params (K, thresholds, chunk size)
    embed.py           — embedding provider abstraction (Ollama or OpenAI, one swap)
    chunk.py           — PDF extraction + overlapping token chunking
    index.py           — ChromaDB upsert/query + BM25 build/load/query
    rewrite.py         — agentic query rewriter (calls Claude via subprocess)
    retrieve.py        — hybrid retrieval per variant + RRF fusion
    rerank.py          — cross-encoder reranker
    verify.py          — citation parser + entailment verifier
    ingest.py          — ingestion pipeline entry point (run manually)
    query.py           — query CLI entry point (called by Claude Code)
    tests/
      test_chunk.py
      test_embed.py
      test_index.py
      test_retrieve.py
      test_rerank.py
      test_verify.py
      conftest.py      — shared fixtures
  CLAUDE.md            — instructs Claude how to use the brain
  requirements.txt
```

---

## Task 1: Project scaffold and config

**Files:**
- Create: `brain/config.py`
- Create: `requirements.txt`
- Create: `brain/tests/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
pymupdf>=1.24.0
tiktoken>=0.7.0
chromadb>=0.5.0
rank_bm25>=0.2.2
sentence-transformers>=3.0.0
requests>=2.31.0
pytest>=8.0.0
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without error. Confirm with `pip show chromadb sentence-transformers`.

- [ ] **Step 3: Create `brain/config.py`**

```python
from pathlib import Path

VAULT_ROOT = Path(__file__).parent.parent
BRAIN_DIR = VAULT_ROOT / "brain"
CHROMA_DIR = BRAIN_DIR / "chroma"
BM25_PATH = BRAIN_DIR / "bm25.pkl"
INGESTION_LOG = BRAIN_DIR / "ingestion_log.json"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVAL = 20      # per retrieval method per variant
TOP_K_RERANK = 8          # final chunks sent to Claude
RRF_K = 60                # RRF constant
RERANK_THRESHOLD = 0.0    # minimum cross-encoder score (0.0 = no filter)
ENTAILMENT_THRESHOLD = 0.3  # minimum score to consider a citation verified

EMBED_PROVIDER = "ollama"   # "ollama" | "openai"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

CHROMA_COLLECTION = "vault"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

- [ ] **Step 4: Create `brain/tests/conftest.py`**

```python
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def sample_pdf(tmp_path):
    """Returns path to a minimal test PDF with known text content."""
    import fitz
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Entity-Relationship Diagrams\nAn ERD models data entities and their relationships.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path

@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "test_p1_c0",
            "course": "IS 6410",
            "filename": "test.pdf",
            "page": 1,
            "slide_title": "Entity-Relationship Diagrams",
            "chunk_index": 0,
            "text": "An ERD models data entities and their relationships.",
        }
    ]
```

- [ ] **Step 5: Verify Python finds the module**

Run: `cd "C:\Users\zwhan\Documents\UofU\Vault\University of Utah - MSIS" && python -c "from brain.config import VAULT_ROOT; print(VAULT_ROOT)"`
Expected: Prints the vault path without error.

- [ ] **Step 6: Commit**

```
git add brain/config.py brain/tests/conftest.py requirements.txt
git commit -m "feat: scaffold brain project with config and test fixtures"
```

---

## Task 2: Embedding provider

**Files:**
- Create: `brain/embed.py`
- Create: `brain/tests/test_embed.py`

- [ ] **Step 1: Write failing test**

```python
# brain/tests/test_embed.py
from unittest.mock import patch, MagicMock
from brain.embed import embed_text, embed_batch

def test_embed_text_returns_list_of_floats():
    fake_vector = [0.1] * 768
    with patch("brain.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embedding": fake_vector},
            raise_for_status=lambda: None,
        )
        result = embed_text("hello world")
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)

def test_embed_batch_returns_list_of_vectors():
    fake_vector = [0.1] * 768
    with patch("brain.embed.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"embedding": fake_vector},
            raise_for_status=lambda: None,
        )
        results = embed_batch(["hello", "world"])
    assert len(results) == 2
    assert len(results[0]) == 768
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `pytest brain/tests/test_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.embed'`

- [ ] **Step 3: Implement `brain/embed.py`**

```python
import os
import requests
from brain.config import EMBED_PROVIDER, OLLAMA_URL, OLLAMA_MODEL, OPENAI_EMBED_MODEL


def embed_text(text: str) -> list[float]:
    if EMBED_PROVIDER == "ollama":
        return _ollama_embed(text)
    return _openai_embed([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    if EMBED_PROVIDER == "ollama":
        return [_ollama_embed(t) for t in texts]
    return _openai_embed(texts)


def _ollama_embed(text: str) -> list[float]:
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


def _openai_embed(texts: list[str]) -> list[list[float]]:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_embed.py -v`
Expected: PASSED (both tests)

- [ ] **Step 5: Commit**

```
git add brain/embed.py brain/tests/test_embed.py
git commit -m "feat: add embedding provider with Ollama/OpenAI swap"
```

---

## Task 3: PDF extraction and chunking

**Files:**
- Create: `brain/chunk.py`
- Create: `brain/tests/test_chunk.py`

- [ ] **Step 1: Write failing tests**

```python
# brain/tests/test_chunk.py
from brain.chunk import extract_pages, chunk_page, build_chunks_from_pdf

def test_extract_pages_returns_page_dicts(sample_pdf):
    pages = extract_pages(sample_pdf, course="IS 6410")
    assert len(pages) >= 1
    assert pages[0]["page"] == 1
    assert "Entity-Relationship" in pages[0]["text"]
    assert pages[0]["course"] == "IS 6410"
    assert pages[0]["filename"] == sample_pdf.name

def test_chunk_page_single_chunk_for_short_text():
    page = {
        "course": "IS 6410",
        "filename": "test.pdf",
        "page": 1,
        "slide_title": "ERDs",
        "text": "Short text that fits in one chunk.",
    }
    chunks = chunk_page(page)
    assert len(chunks) == 1
    assert chunks[0]["id"] == "IS 6410_test.pdf_p1_c0"
    assert chunks[0]["text"] == "Short text that fits in one chunk."

def test_chunk_page_overlaps_long_text():
    long_text = "word " * 600  # ~600 tokens
    page = {
        "course": "IS 6410", "filename": "test.pdf",
        "page": 2, "slide_title": "Long Slide", "text": long_text,
    }
    chunks = chunk_page(page)
    assert len(chunks) >= 2
    # overlap: last tokens of chunk 0 appear at start of chunk 1
    words_0 = chunks[0]["text"].split()
    words_1 = chunks[1]["text"].split()
    assert words_0[-1] == words_1[0] or words_0[-10:] == words_1[:10]

def test_build_chunks_from_pdf(sample_pdf):
    chunks = build_chunks_from_pdf(sample_pdf, course="IS 6410")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c and "page" in c for c in chunks)
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest brain/tests/test_chunk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.chunk'`

- [ ] **Step 3: Implement `brain/chunk.py`**

```python
import re
from pathlib import Path
import fitz  # pymupdf
import tiktoken
from brain.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def extract_pages(pdf_path: Path, course: str) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    outline = {p: title for title, _, p in _extract_outline(doc)}
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        pages.append({
            "course": course,
            "filename": pdf_path.name,
            "page": i + 1,
            "slide_title": outline.get(i, ""),
            "text": text,
        })
    doc.close()
    return pages


def _extract_outline(doc) -> list[tuple]:
    try:
        return [(title, _, page) for title, _, page, *_ in doc.get_toc()]
    except Exception:
        return []


def chunk_page(page: dict) -> list[dict]:
    tokens = _enc.encode(page["text"])
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_text = _enc.decode(tokens[start:end])
        chunk_id = f"{page['course']}_{page['filename']}_p{page['page']}_c{idx}"
        chunks.append({
            "id": chunk_id,
            "course": page["course"],
            "filename": page["filename"],
            "page": page["page"],
            "slide_title": page["slide_title"],
            "chunk_index": idx,
            "text": chunk_text,
        })
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP_TOKENS
        idx += 1
    return chunks


def build_chunks_from_pdf(pdf_path: Path, course: str) -> list[dict]:
    pages = extract_pages(pdf_path, course)
    chunks = []
    for page in pages:
        chunks.extend(chunk_page(page))
    return chunks
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_chunk.py -v`
Expected: All 4 tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/chunk.py brain/tests/test_chunk.py
git commit -m "feat: PDF extraction and overlapping token chunking"
```

---

## Task 4: Dual index (ChromaDB + BM25)

**Files:**
- Create: `brain/index.py`
- Create: `brain/tests/test_index.py`

- [ ] **Step 1: Write failing tests**

```python
# brain/tests/test_index.py
import pytest
from unittest.mock import patch, MagicMock
from brain.index import (
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
    bm25_path = tmp_path / "bm25.pkl"
    index, corpus = build_bm25(sample_chunks, bm25_path)
    results = query_bm25(index, corpus, sample_chunks, "ERD entities relationships", top_k=5)
    assert len(results) >= 1
    assert results[0]["id"] == "IS 6410_test.pdf_p1_c0"

def test_load_bm25_roundtrip(tmp_path, sample_chunks):
    bm25_path = tmp_path / "bm25.pkl"
    build_bm25(sample_chunks, bm25_path)
    loaded_index, loaded_corpus, loaded_chunks = load_bm25(bm25_path)
    results = query_bm25(loaded_index, loaded_corpus, loaded_chunks, "entities", top_k=5)
    assert len(results) >= 1
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest brain/tests/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.index'`

- [ ] **Step 3: Implement `brain/index.py`**

```python
import pickle
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from brain.config import CHROMA_DIR, CHROMA_COLLECTION


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(CHROMA_COLLECTION)


def upsert_chunks(collection, chunks: list[dict], embeddings: list[list[float]]):
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{k: v for k, v in c.items() if k not in ("text", "id")} for c in chunks],
    )


def query_dense(collection, query_vector: list[float], top_k: int) -> list[dict]:
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    output = []
    for i, chunk_id in enumerate(results["ids"][0]):
        output.append({
            "id": chunk_id,
            "text": results["documents"][0][i],
            "score": 1.0 - results["distances"][0][i],  # cosine distance → similarity
            **results["metadatas"][0][i],
        })
    return output


def build_bm25(chunks: list[dict], bm25_path: Path):
    tokenized = [c["text"].lower().split() for c in chunks]
    index = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump((index, tokenized, chunks), f)
    return index, tokenized


def load_bm25(bm25_path: Path):
    with open(bm25_path, "rb") as f:
        return pickle.load(f)  # (index, tokenized_corpus, chunks)


def query_bm25(index, corpus, chunks: list[dict], query: str, top_k: int) -> list[dict]:
    tokens = query.lower().split()
    scores = index.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"id": chunks[i]["id"], "text": chunks[i]["text"], "score": float(s), **{k: v for k, v in chunks[i].items() if k not in ("id", "text")}} for i, s in ranked if s > 0]
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_index.py -v`
Expected: All 3 tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/index.py brain/tests/test_index.py
git commit -m "feat: ChromaDB dense index and BM25 sparse index with upsert/query"
```

---

## Task 5: Ingestion pipeline

**Files:**
- Create: `brain/ingest.py`

- [ ] **Step 1: Write a smoke test**

```python
# brain/tests/test_ingest.py
import json
from unittest.mock import patch, MagicMock
from brain.ingest import find_unindexed_pdfs, log_indexed, load_log

def test_find_unindexed_pdfs(tmp_path):
    (tmp_path / "course").mkdir()
    pdf1 = tmp_path / "course" / "slides.pdf"
    pdf1.touch()
    log = {}
    result = find_unindexed_pdfs(tmp_path, log)
    assert pdf1 in result

def test_already_indexed_pdf_excluded(tmp_path):
    pdf1 = tmp_path / "slides.pdf"
    pdf1.touch()
    log = {str(pdf1): "done"}
    result = find_unindexed_pdfs(tmp_path, log)
    assert pdf1 not in result

def test_log_roundtrip(tmp_path):
    log_path = tmp_path / "log.json"
    log = {}
    log_indexed(log, tmp_path / "a.pdf", log_path)
    loaded = load_log(log_path)
    assert str(tmp_path / "a.pdf") in loaded
```

- [ ] **Step 2: Run to confirm it fails**

Run: `pytest brain/tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.ingest'`

- [ ] **Step 3: Implement `brain/ingest.py`**

```python
import json
import sys
from pathlib import Path
from brain.config import VAULT_ROOT, INGESTION_LOG, BM25_PATH, TOP_K_RETRIEVAL
from brain.chunk import build_chunks_from_pdf
from brain.embed import embed_batch
from brain.index import get_collection, upsert_chunks, build_bm25, load_bm25


def load_log(log_path: Path) -> dict:
    if log_path.exists():
        return json.loads(log_path.read_text())
    return {}


def log_indexed(log: dict, pdf_path: Path, log_path: Path):
    log[str(pdf_path)] = "done"
    log_path.write_text(json.dumps(log, indent=2))


def find_unindexed_pdfs(root: Path, log: dict) -> list[Path]:
    return [p for p in root.rglob("*.pdf") if str(p) not in log]


def _course_from_path(pdf_path: Path) -> str:
    """Infer course name from parent folder structure."""
    parts = pdf_path.relative_to(VAULT_ROOT).parts
    return parts[1] if len(parts) > 2 else parts[0]


def run_ingestion():
    log = load_log(INGESTION_LOG)
    pdfs = find_unindexed_pdfs(VAULT_ROOT, log)
    if not pdfs:
        print("Nothing to index.")
        return

    collection = get_collection()

    # Load existing BM25 chunks to append to
    if BM25_PATH.exists():
        _, _, existing_chunks = load_bm25(BM25_PATH)
    else:
        existing_chunks = []

    all_new_chunks = []
    for pdf_path in pdfs:
        course = _course_from_path(pdf_path)
        print(f"Indexing {pdf_path.name} ({course})...")
        chunks = build_chunks_from_pdf(pdf_path, course)
        vectors = embed_batch([c["text"] for c in chunks])
        upsert_chunks(collection, chunks, vectors)
        all_new_chunks.extend(chunks)
        log_indexed(log, pdf_path, INGESTION_LOG)
        print(f"  {len(chunks)} chunks indexed.")

    build_bm25(existing_chunks + all_new_chunks, BM25_PATH)
    print(f"Done. BM25 index rebuilt with {len(existing_chunks) + len(all_new_chunks)} total chunks.")


if __name__ == "__main__":
    run_ingestion()
```

- [ ] **Step 4: Run smoke tests**

Run: `pytest brain/tests/test_ingest.py -v`
Expected: All 3 tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/ingest.py brain/tests/test_ingest.py
git commit -m "feat: ingestion pipeline with incremental indexing and log tracking"
```

---

## Task 6: Agentic query rewriter

**Files:**
- Create: `brain/rewrite.py`
- Create: `brain/tests/test_rewrite.py`

- [ ] **Step 1: Write failing test**

```python
# brain/tests/test_rewrite.py
from unittest.mock import patch
from brain.rewrite import rewrite_query, parse_variants

def test_parse_variants_extracts_numbered_list():
    raw = "1. What is an ERD?\n2. Entity relationship diagram definition\n3. ERD data modeling"
    variants = parse_variants(raw)
    assert len(variants) == 3
    assert "ERD" in variants[0]

def test_parse_variants_handles_bullet_list():
    raw = "- ERD definition\n- entity relationship diagram\n- data modeling entities"
    variants = parse_variants(raw)
    assert len(variants) == 3

def test_rewrite_query_returns_list_of_strings():
    with patch("brain.rewrite.subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1. ERD definition\n2. entity relationship diagram\n3. data modeling"
        mock_run.return_value.returncode = 0
        variants = rewrite_query("What is an ERD?")
    assert isinstance(variants, list)
    assert len(variants) >= 1
    assert all(isinstance(v, str) for v in variants)
    # original query always included
    assert "What is an ERD?" in variants
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest brain/tests/test_rewrite.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.rewrite'`

- [ ] **Step 3: Implement `brain/rewrite.py`**

```python
import re
import subprocess
import sys

REWRITE_PROMPT = """\
You are a query rewriting assistant for a student's academic knowledge base.

Given the user's question, produce exactly 4 rewritten variants as a numbered list.
Apply these strategies:
1. Decompose multi-part questions into the most important sub-question
2. Expand acronyms and abbreviations
3. Add domain synonyms (e.g. "ERD" → "entity relationship diagram")
4. Write a short hypothetical answer fragment (HyDE) to improve retrieval

Output ONLY the 4 numbered variants, nothing else.

User question: {question}
"""


def rewrite_query(question: str) -> list[str]:
    prompt = REWRITE_PROMPT.format(question=question)
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return [question]  # fallback: use original
    variants = parse_variants(result.stdout)
    # always include original to guarantee coverage
    if question not in variants:
        variants.insert(0, question)
    return variants


def parse_variants(raw: str) -> list[str]:
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    cleaned = []
    for line in lines:
        # strip leading "1." or "-" or "*"
        line = re.sub(r"^(\d+\.\s+|[-*]\s+)", "", line)
        if line:
            cleaned.append(line)
    return cleaned
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_rewrite.py -v`
Expected: All 3 tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/rewrite.py brain/tests/test_rewrite.py
git commit -m "feat: agentic query rewriter with Claude subprocess and HyDE"
```

---

## Task 7: Hybrid retrieval with RRF fusion

**Files:**
- Create: `brain/retrieve.py`
- Create: `brain/tests/test_retrieve.py`

- [ ] **Step 1: Write failing tests**

```python
# brain/tests/test_retrieve.py
from brain.retrieve import reciprocal_rank_fusion, deduplicate_by_id

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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest brain/tests/test_retrieve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.retrieve'`

- [ ] **Step 3: Implement `brain/retrieve.py`**

```python
from collections import defaultdict
from brain.config import RRF_K, TOP_K_RETRIEVAL
from brain.embed import embed_text
from brain.index import get_collection, query_dense, load_bm25, query_bm25
from brain.config import BM25_PATH


def reciprocal_rank_fusion(ranked_lists: list[list[dict]]) -> list[dict]:
    scores = defaultdict(float)
    chunk_store = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk["id"]] += 1.0 / (RRF_K + rank)
            chunk_store[chunk["id"]] = chunk
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for cid in sorted_ids:
        c = dict(chunk_store[cid])
        c["rrf_score"] = scores[cid]
        result.append(c)
    return result


def deduplicate_by_id(chunks: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            result.append(c)
    return result


def hybrid_retrieve(query_variants: list[str], top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    collection = get_collection()
    bm25_index, bm25_corpus, bm25_chunks = load_bm25(BM25_PATH)

    all_ranked_lists = []
    for variant in query_variants:
        vec = embed_text(variant)
        dense_results = query_dense(collection, vec, top_k)
        sparse_results = query_bm25(bm25_index, bm25_corpus, bm25_chunks, variant, top_k)
        all_ranked_lists.extend([dense_results, sparse_results])

    fused = reciprocal_rank_fusion(all_ranked_lists)
    return deduplicate_by_id(fused)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_retrieve.py -v`
Expected: All 3 tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/retrieve.py brain/tests/test_retrieve.py
git commit -m "feat: hybrid retrieval with RRF fusion across query variants"
```

---

## Task 8: Cross-encoder reranker

**Files:**
- Create: `brain/rerank.py`
- Create: `brain/tests/test_rerank.py`

- [ ] **Step 1: Write failing test**

```python
# brain/tests/test_rerank.py
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest brain/tests/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.rerank'`

- [ ] **Step 3: Implement `brain/rerank.py`**

```python
from functools import lru_cache
from sentence_transformers import CrossEncoder
from brain.config import CROSS_ENCODER_MODEL, TOP_K_RERANK


@lru_cache(maxsize=1)
def _get_model():
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank_chunks(query: str, chunks: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_rerank.py -v`
Expected: Both tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/rerank.py brain/tests/test_rerank.py
git commit -m "feat: cross-encoder reranker with cached model loading"
```

---

## Task 9: Citation verifier

**Files:**
- Create: `brain/verify.py`
- Create: `brain/tests/test_verify.py`

- [ ] **Step 1: Write failing tests**

```python
# brain/tests/test_verify.py
from brain.verify import parse_citations, format_verified_response, VerifiedClaim

def test_parse_citations_extracts_inline_sources():
    response = (
        "An ERD models entities and relationships [source: week3.pdf, page 12]. "
        "Normalization reduces redundancy [source: week4.pdf, page 5]."
    )
    claims = parse_citations(response)
    assert len(claims) == 2
    assert claims[0].filename == "week3.pdf"
    assert claims[0].page == 12
    assert "ERD models entities" in claims[0].claim_text

def test_parse_citations_no_sources_returns_uncited():
    response = "This claim has no citation."
    claims = parse_citations(response)
    assert len(claims) == 1
    assert claims[0].filename is None

def test_format_verified_response_shows_grounding_ratio():
    claims = [
        VerifiedClaim("Claim A", "a.pdf", 1, "chunk text", True),
        VerifiedClaim("Claim B", "b.pdf", 2, "chunk text", False),
    ]
    output = format_verified_response(claims)
    assert "1/2 claims verified" in output
    assert "⚠ unverified" in output
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest brain/tests/test_verify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'brain.verify'`

- [ ] **Step 3: Implement `brain/verify.py`**

```python
import re
from dataclasses import dataclass
from brain.config import ENTAILMENT_THRESHOLD
from brain.rerank import _get_model


@dataclass
class VerifiedClaim:
    claim_text: str
    filename: str | None
    page: int | None
    chunk_text: str | None
    verified: bool


_CITATION_RE = re.compile(
    r"([^[]+?)\s*\[source:\s*([^,\]]+),\s*page\s*(\d+)\]",
    re.IGNORECASE,
)


def parse_citations(response: str) -> list[VerifiedClaim]:
    claims = []
    matches = list(_CITATION_RE.finditer(response))
    if not matches:
        # treat entire response as one uncited claim
        return [VerifiedClaim(response.strip(), None, None, None, False)]
    for m in matches:
        claims.append(VerifiedClaim(
            claim_text=m.group(1).strip(),
            filename=m.group(2).strip(),
            page=int(m.group(3)),
            chunk_text=None,
            verified=False,
        ))
    return claims


def verify_claims(claims: list[VerifiedClaim], chunks: list[dict]) -> list[VerifiedClaim]:
    chunk_map = {(c.get("filename", ""), c.get("page", 0)): c["text"] for c in chunks}
    model = _get_model()
    for claim in claims:
        if claim.filename is None:
            continue
        chunk_text = chunk_map.get((claim.filename, claim.page))
        if not chunk_text:
            claim.verified = False
            continue
        claim.chunk_text = chunk_text
        score = float(model.predict([(claim.claim_text, chunk_text)]))
        claim.verified = score >= ENTAILMENT_THRESHOLD
    return claims


def format_verified_response(claims: list[VerifiedClaim]) -> str:
    lines = []
    verified_count = sum(1 for c in claims if c.verified)
    for claim in claims:
        if claim.filename is None:
            lines.append(f"{claim.claim_text} ⚠ unverified (no citation)")
        elif claim.verified:
            lines.append(f"{claim.claim_text} [source: {claim.filename}, page {claim.page}]")
        else:
            lines.append(f"{claim.claim_text} [source: {claim.filename}, page {claim.page}] ⚠ unverified")
    lines.append(f"\nGrounding: {verified_count}/{len(claims)} claims verified")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest brain/tests/test_verify.py -v`
Expected: All 3 tests PASSED

- [ ] **Step 5: Commit**

```
git add brain/verify.py brain/tests/test_verify.py
git commit -m "feat: citation parser and entailment-based claim verifier"
```

---

## Task 10: Query CLI entry point

**Files:**
- Create: `brain/query.py`

- [ ] **Step 1: Implement `brain/query.py`**

```python
"""
Usage: python brain/query.py "Your question here"

Runs the full pipeline: rewrite → hybrid retrieve → rerank → format context.
Output is a context block for Claude to consume. Does NOT call Claude itself.
"""
import sys
from brain.rewrite import rewrite_query
from brain.retrieve import hybrid_retrieve
from brain.rerank import rerank_chunks
from brain.config import TOP_K_RERANK


def run_query(question: str) -> str:
    print(f"[brain] Rewriting query...", file=sys.stderr)
    variants = rewrite_query(question)
    print(f"[brain] {len(variants)} variants generated.", file=sys.stderr)

    print(f"[brain] Running hybrid retrieval...", file=sys.stderr)
    candidates = hybrid_retrieve(variants)
    print(f"[brain] {len(candidates)} candidates before reranking.", file=sys.stderr)

    print(f"[brain] Reranking...", file=sys.stderr)
    top_chunks = rerank_chunks(question, candidates, top_k=TOP_K_RERANK)

    lines = [f"# Retrieved context for: {question}\n"]
    for i, chunk in enumerate(top_chunks, 1):
        lines.append(
            f"## [{i}] {chunk.get('filename', 'unknown')}, page {chunk.get('page', '?')} "
            f"(course: {chunk.get('course', '?')})\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python brain/query.py \"Your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(run_query(question))
```

- [ ] **Step 2: Smoke test with a dry run (no real index needed)**

Run: `python brain/query.py "test" 2>&1 | head -5`
Expected: Prints `[brain] Rewriting query...` before hitting any real index errors. (Full end-to-end requires Ollama running and slides ingested.)

- [ ] **Step 3: Commit**

```
git add brain/query.py
git commit -m "feat: query CLI entry point — rewrite → retrieve → rerank → format"
```

---

## Task 11: CLAUDE.md integration instructions

**Files:**
- Modify: `University of Utah - MSIS/CLAUDE.md`

- [ ] **Step 1: Write `CLAUDE.md`**

```markdown
# Second Brain — Claude Code Instructions

## How to answer study questions

When the user asks a question about course material, you MUST:

1. Run the retrieval pipeline first:
   ```
   python brain/query.py "<user's question>"
   ```
2. Use ONLY the returned context chunks as your factual basis.
3. Tag every factual claim with an inline citation: `[source: filename, page N]`
4. If retrieval returns no results, respond: "I don't have slides covering this topic."
5. Never assert facts not present in the retrieved chunks.

## Citation rules

- Every sentence containing a factual claim needs `[source: filename, page N]`
- Synthesis across multiple chunks is allowed — cite all sources used
- Your own explanations and analogies are fine — label them clearly as "explanation:" not as facts

## Adding new slides

When the user adds new PDFs to the vault:
```
python brain/ingest.py
```

## Running tests

```
pytest brain/tests/ -v
```
```

- [ ] **Step 2: Verify the file is readable**

Run: `python -c "from pathlib import Path; print(Path('CLAUDE.md').read_text()[:100])"`
Expected: Prints first 100 chars of the file.

- [ ] **Step 3: Commit**

```
git add CLAUDE.md
git commit -m "feat: CLAUDE.md instructs Claude to use retrieval pipeline before answering"
```

---

## Task 12: End-to-end smoke test

This task runs the full pipeline with real slides. Requires Ollama running with `nomic-embed-text` pulled.

- [ ] **Step 1: Start Ollama and pull embedding model**

Run: `ollama pull nomic-embed-text`
Expected: Model downloads (274MB, one-time).

- [ ] **Step 2: Ingest one course folder**

Run: `python brain/ingest.py`
Expected: Prints progress per PDF, ends with BM25 rebuild confirmation.

- [ ] **Step 3: Run a real query**

Run: `python brain/query.py "What is normalization in databases?"`
Expected: Prints a context block with 8 chunks, each with filename and page number.

- [ ] **Step 4: Run full test suite**

Run: `pytest brain/tests/ -v`
Expected: All tests PASSED (unit tests only — no live Ollama required for tests)

- [ ] **Step 5: Final commit**

```
git add -A
git commit -m "feat: complete second brain RAG pipeline — end-to-end verified"
```
