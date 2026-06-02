# Second Brain — MSIS Coursework RAG System

A personal second brain for MSIS coursework. PDFs, docs, and notes are automatically chunked, embedded, and indexed. Ask questions in Claude Code and get cited answers pulled directly from your course material. A semantic graph shows connections across courses.

## How it works

1. **Ingest** — course materials (PDF, DOCX, TXT, MD) are extracted, chunked, and stored in a dual index: ChromaDB for semantic search and BM25 for keyword search
2. **Query** — questions are rewritten into multiple variants by Claude, retrieved via hybrid search, reranked by a cross-encoder, and answered with inline citations
3. **Verify** — every citation is checked using an NLI entailment model to confirm the source actually supports the claim
4. **Graph** — an Obsidian graph view shows semantic connections across documents, built using mutual top-K similarity

## Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), Claude Code CLI

```powershell
# Install dependencies
uv sync --extra dev

# Pull the embedding model (one-time, ~274MB)
ollama pull nomic-embed-text
```

## Usage

**Index your course materials:**
```powershell
# Place files in raw/<course-name>/
uv run python brain/ingest.py
uv run python brain/graph.py
```

**Ask questions** (in Claude Code, from the vault directory):
```
What are the five Scrum events?
```
Claude automatically runs the retrieval pipeline before answering.

**Run tests:**
```powershell
uv run pytest brain/tests/ -v
```

## File structure

```
raw/                    # source documents (gitignored PDFs)
  IS 6410/
  OSC 6680/
  ...
notes/                  # generated Obsidian graph notes
  IS 6410/
  OSC 6680/
  ...
brain/
  ingest.py             # index new documents
  query.py              # retrieval pipeline CLI
  graph.py              # Obsidian graph generator
  chunk.py              # PDF/DOCX/TXT/MD extraction and chunking
  embed.py              # Ollama / OpenAI embedding provider
  index.py              # ChromaDB + BM25 index
  retrieve.py           # hybrid retrieval + RRF fusion
  rerank.py             # cross-encoder reranker
  verify.py             # NLI citation verifier
  rewrite.py            # agentic query rewriter
  config.py             # all tunable parameters
  tests/
docs/
  future-ideas.md       # planned features
  superpowers/
    specs/              # design documents
    plans/              # implementation plans
CLAUDE.md               # instructs Claude Code how to use this system
pyproject.toml
```

## Configuration

All tunable parameters are in `brain/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `GRAPH_MIN_SIMILARITY` | 0.72 | Minimum similarity score for graph links |
| `GRAPH_TOP_K` | 10 | Candidate pool size for mutual top-K filtering |
| `TOP_K_RERANK` | 8 | Chunks returned to Claude per query |
| `ENTAILMENT_THRESHOLD` | 0.5 | Minimum NLI probability to verify a citation |
| `SLIDE_PAGE_TOKEN_THRESHOLD` | 150 | Avg tokens/page below this → slide deck chunking |
| `EMBED_PROVIDER` | `"ollama"` | `"ollama"` or `"openai"` |
