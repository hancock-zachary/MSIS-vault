---
name: querying-second-brain
description: Answers factual study questions using the MSIS vault. Runs the full retrieval pipeline — wiki index, wiki pages, then query.py — and returns cited answers. Use when the user asks ANY factual question about course material. NEVER modifies files.
---

# Querying the Second Brain

## Hard constraints

- **Read-only.** Never create, edit, or delete any file. No exceptions.
- **Every factual claim must be cited**: `[source: filename, page N]`
- **No filing back.** Do not offer to save the answer to the wiki.
- If no relevant content exists: "I don't have citable information covering this topic."

## Pipeline — always run all three steps in order

### Step 1 — Navigate: read `wiki/index.md`
Identify which documents are relevant to the question. The index lists every document by course with a one-line preview.

### Step 2 — Context: read relevant wiki pages
Each wiki page is in `wiki/<Course>/<Title>.md`. Read the pages identified in Step 1. This gives document-level summaries, top excerpts, and cross-references to related pages. Follow any `[[wikilinks]]` that point to additional relevant pages.

### Step 3 — Retrieve: run query.py
```bash
uv run python src/query.py "<question>"
```
This runs the full pipeline (rewrite → hybrid retrieve → rerank) and returns the top 8 chunks most relevant to the question. Use these chunks as the primary factual source.

## Answering

- Use Step 3 chunks as the primary evidence. Use Step 2 wiki pages for context and cross-references.
- Cite every factual sentence: `[source: filename, page N]`
- Synthesis across multiple sources is allowed — cite all sources used.
- Your own explanations and analogies are fine — label them `(explanation)`.
- Chunks with `is_stub: true` indicate unreadable content (diagrams, scanned images). Do not cite stubs as factual sources — they only confirm a topic exists in that document.
