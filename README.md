# Second Brain — MSIS Coursework RAG System

A personal second brain for MSIS coursework. PDFs, docs, and notes are automatically chunked, embedded, and indexed. Ask questions in Claude Code and get cited answers pulled directly from your course material. A semantic graph shows connections across courses.

## How it works

1. **Ingest** — course materials (PDF, DOCX, TXT, MD) are extracted, then routed to the right chunking strategy based on document signals: `slides` (low token density), `structured` (headings/DOCX/MD), `semantic` (split on embedding cosine drops), or `window` (fallback). Garbled pages (scanned images) get a salvage pass that extracts the page title as a stub chunk. Chunks are stored in a dual index: ChromaDB for semantic search and BM25 for keyword search.
2. **Query** — questions are rewritten into 5 variants by Claude (keywords, paraphrase, HyDE hypothetical answer, specific, broader), retrieved via hybrid search with RRF fusion, then reranked by a cross-encoder
3. **Verify** — every citation is checked using an NLI entailment model (`cross-encoder/nli-deberta-v3-small`) to confirm the source actually supports the claim
4. **Graph** — an Obsidian wiki in `wiki/` stores key excerpts and mutual top-K wikilinks across documents; `graph.py` rebuilds it from the index and cleans up stale pages

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
uv run python src/ingest.py
uv run python src/graph.py
```

**Ask questions** (in Claude Code, from the vault directory):
```
What are the five Scrum events?
```
Claude automatically runs the retrieval pipeline before answering.

**Run tests:**
```powershell
uv run pytest src/tests/ -v
```

## File structure

```
raw/                    # source documents (gitignored PDFs)
  IS 6410/
  OSC 6680/
  ...
wiki/                  # generated Obsidian graph notes
  IS 6410/
  OSC 6680/
  ...
src/
  ingest.py             # index new documents
  query.py              # retrieval pipeline CLI
  graph.py              # Obsidian graph generator
  chunk.py              # PDF/DOCX/TXT/MD extraction and chunking strategies
  router.py             # signal-based chunking router (slides/semantic/structured/window)
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
CLAUDE.md               # instructs the AI assistant how to use this system
AGENTS.md               # same file, for Codex (OpenAI)
GEMINI.md               # same file, for Gemini CLI
pyproject.toml
```

## Using with other AI assistants

The retrieval pipeline is provider-agnostic. To use with Gemini CLI or Codex instead of Claude Code, just copy the instruction file under the name your tool expects:

| Tool | File to create |
|---|---|
| Claude Code | `CLAUDE.md` (already exists) |
| Gemini CLI | `GEMINI.md` |
| Codex (OpenAI) | `AGENTS.md` |

```powershell
# For Gemini CLI
Copy-Item CLAUDE.md GEMINI.md

# For Codex
Copy-Item CLAUDE.md AGENTS.md
```

The content is identical — the instructions are plain text that any AI coding assistant will follow. No other changes are needed.

## Configuration

All tunable parameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `GRAPH_MIN_SIMILARITY` | 0.72 | Minimum similarity score for graph links |
| `GRAPH_TOP_K` | 10 | Candidate pool size for mutual top-K filtering |
| `TOP_K_RERANK` | 8 | Chunks returned to Claude per query |
| `ENTAILMENT_THRESHOLD` | 0.5 | Minimum NLI probability to verify a citation |
| `SLIDE_PAGE_TOKEN_THRESHOLD` | 150 | Avg tokens/page below this → slide deck chunking |
| `EMBED_PROVIDER` | `"ollama"` | `"ollama"` or `"openai"` |
