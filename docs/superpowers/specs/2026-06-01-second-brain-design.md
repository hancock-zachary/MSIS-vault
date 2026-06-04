# Second Brain — Professional RAG System Design

**Date:** 2026-06-01
**Context:** Obsidian vault for MSIS program at University of Utah. Primary source material is PDF lecture slides. Goal: a queryable, citation-verified knowledge base that supports deep understanding and lasting knowledge organization.

---

## 1. Goals

- Make all course PDF slides queryable via natural language
- Retrieve answers that are semantically precise, not just keyword-matched
- Ensure every factual claim is grounded in and traceable to a specific slide and page
- Support two entry points: Claude Code CLI and Obsidian sidebar (via Smart Connections or future plugin)
- Build in Python as a learning project — no black boxes

---

## 2. Architecture Overview

```
PDFs (courses/)
    │
    ▼
[Ingestion Pipeline]
    │  pymupdf extraction → chunking → dual indexing
    │
    ├──▶ ChromaDB (dense vectors, nomic-embed-text via Ollama)
    └──▶ BM25 index (sparse keyword, rank_bm25, pickled)
    │
    ▼
[Graph Generator]
    │  document-level embeddings → similarity → Obsidian wikilinks
    └──▶ wiki/ (one .md per PDF, [[wikilinks]] to top-5 related docs)
    
Query
    │
    ▼
[Agentic Query Rewriter]
    │  Claude rewrites question into 3-5 variants
    │
    ▼
[Hybrid Retrieval]
    │  Dense + sparse retrieval per variant → RRF fusion → deduplicate
    │
    ▼
[Cross-Encoder Reranker]
    │  cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free)
    │
    ▼
[Claude Synthesis]
    │  Prompted to tag every claim with inline [source: file, page N]
    │
    ▼
[Citation Verifier]
    │  Checks chunk exists + entailment score per claim
    │  Flags unverified claims with ⚠, reports grounding ratio
    │
    ▼
Answer to user
```

---

## 3. Ingestion Pipeline

**Trigger:** Manual — run `ingest.py` when new slides are added to the vault.

**Steps:**
1. Walk vault directory, find all `.pdf` files not yet indexed (tracked by a `ingestion_log.json`)
2. Extract text per page using `pymupdf` (`fitz`). Preserve page number and attempt to extract slide title from PDF outline/bookmarks.
3. Chunk each page into overlapping 500-token windows (overlap: 50 tokens) using `tiktoken` for accurate token counting.
4. Embed each chunk with `nomic-embed-text` via Ollama HTTP API (local, free).
5. Upsert into ChromaDB with metadata: `{course, filename, page, slide_title, chunk_index}`.
6. Rebuild BM25 index from all chunks and pickle to `src/bm25.pkl`.

**Chunk metadata schema:**
```json
{
  "id": "IS6410_week3_slides_p12_c0",
  "course": "IS 6410",
  "filename": "week3_slides.pdf",
  "page": 12,
  "slide_title": "Entity-Relationship Diagrams",
  "chunk_index": 0,
  "text": "..."
}
```

---

## 4. Agentic Query Rewriting

Before retrieval, Claude is called with a structured prompt to decompose and expand the user's question:

**Input:** Raw user question
**Output:** List of 3-5 rewritten variants

**Rewriting strategies applied:**
- Decompose multi-part questions into sub-questions
- Expand acronyms and course-specific terms
- Add domain synonyms (e.g., "ERD" → "entity relationship diagram")
- Generate a hypothetical answer fragment (HyDE technique) to improve dense retrieval

Each variant is run through hybrid retrieval independently. Results are merged before reranking.

---

## 5. Hybrid Retrieval

For each query variant:
- **Dense:** Top-20 from ChromaDB cosine similarity
- **Sparse:** Top-20 from BM25

Results per variant are merged using **Reciprocal Rank Fusion (RRF)**:
```
RRF_score(chunk) = Σ 1 / (k + rank_i)   where k=60
```

After RRF fusion across all variants, deduplicate by chunk ID, keeping highest score. Carry forward top-50 candidates to reranker.

---

## 6. Cross-Encoder Reranking

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, runs locally, ~25MB)

- Scores each (original question, chunk) pair jointly — significantly more accurate than embedding similarity
- Sort by score descending, take top-K (default K=8) for synthesis

---

## 7. Citation Certification

**Synthesis prompt constraint:** Claude is instructed via system prompt that:
- Every factual claim must be followed by `[source: filename, page N]`
- If a claim cannot be sourced from retrieved chunks, it must not be stated
- If retrieval returns zero results, respond: "I don't have slides covering this topic."

**Verification pass (post-generation):**
1. Parse Claude's response to extract (claim, cited chunk ID) pairs
2. Verify chunk ID exists in ChromaDB
3. Score semantic entailment: cross-encoder score of (claim, chunk text) — threshold 0.3
4. Claims below threshold flagged inline with `⚠ unverified`
5. Final line of every response reports: `Grounding: 12/13 claims verified`

---

## 8. Claude Code Integration

**`CLAUDE.md` instructions added to vault root:**
- When answering study questions, invoke `src/query.py "<question>"` first
- Use returned chunks as the only factual basis for the answer
- Follow citation and grounding constraints

**`src/query.py` CLI:**
```
python src/query.py "What is the difference between OLTP and OLAP?"
```
Returns formatted context block with top-K chunks and metadata, ready for Claude to consume.

---

## 9. File Structure

```
Vault/
  University of Utah - MSIS/
    courses/
      IS 6410/
        slides/          ← PDFs live here
        readings/
        additional_files/
      IS 6495/
        ...
    wiki/               ← generated Obsidian notes (one per PDF, with wikilinks)
    src/
      ingest.py          ← ingestion pipeline
      graph.py           ← Obsidian graph note generator
      query.py           ← query CLI
      rewrite.py         ← agentic query rewriter
      rerank.py          ← cross-encoder reranker
      verify.py          ← citation verifier
      config.py          ← all tunable constants
      embed.py           ← embedding provider (Ollama / OpenAI)
      chunk.py           ← PDF extraction and chunking
      index.py           ← ChromaDB + BM25 index management
      retrieve.py        ← hybrid retrieval + RRF fusion
      chroma/            ← ChromaDB persistence (gitignored)
      bm25.pkl           ← BM25 index (gitignored)
      ingestion_log.json ← tracks indexed files (gitignored)
    CLAUDE.md
    pyproject.toml
```

---

## 10. Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` | PDF text extraction |
| `tiktoken` | Accurate token counting for chunking |
| `chromadb` | Vector store |
| `rank_bm25` | BM25 sparse index |
| `sentence-transformers` | Cross-encoder reranker |
| `requests` | Ollama HTTP API calls |
| Ollama + `nomic-embed-text` | Local embeddings (free) |

> **Fallback option:** If Ollama is too cumbersome on low-powered hardware, the embedding provider can be swapped to the OpenAI API (`text-embedding-3-small`). Cost is ~$0.02 per million tokens — negligible for a student vault. The provider is isolated behind a single function so the swap requires no other changes to the pipeline.

---

## 11. Graph View Generation

`src/graph.py` generates one Obsidian markdown note per indexed PDF and populates it with `[[wikilinks]]` to the most semantically similar other documents. This makes Obsidian's graph view reflect genuine conceptual relationships across all courses.

**How it works:**
1. For each PDF in `ingestion_log.json`, fetch all its chunk embeddings from ChromaDB
2. Average them into a single document-level embedding
3. Query ChromaDB for the top-50 most similar chunks from *other* files
4. Deduplicate to unique filenames, take the top `GRAPH_TOP_K` (default: 5)
5. Write a markdown note to `wiki/<title>.md` with metadata and wikilinks

**Usage:**
```
uv run python src/ingest.py   # index PDFs first
uv run python src/graph.py   # then generate notes
```

Re-running `graph.py` is safe — notes are overwritten with fresh similarity data each time.

---

## 12. Out of Scope (for now)

- Obsidian plugin UI (future phase)
- Automatic ingestion on file save
- Multi-modal: slide images/diagrams
- Spaced repetition / flashcard generation (future phase)
