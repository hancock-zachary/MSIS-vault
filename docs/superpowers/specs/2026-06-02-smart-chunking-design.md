# Smart Chunking — Signal-Based Router Design

**Date:** 2026-06-02
**Context:** Upgrade to the ingestion pipeline. Replaces the single-strategy chunking approach with a signal-based router that selects the optimal chunking strategy per document, attaches a `ChunkingProfile` to every chunk as metadata, and salvages title text from garbled image pages rather than discarding them entirely.

---

## 1. Goals

- Select the best chunking strategy for each document based on measurable signals, not file type alone
- Expose every routing decision as chunk metadata so failures are visible and debuggable
- Salvage topic information from pages whose body is garbled (diagrams, scanned images) by extracting title text as stub chunks
- Flag stub chunks so the retrieval and citation layers handle them correctly
- Preserve page-level citation accuracy throughout

---

## 2. Architecture

Three files change, one new file is added:

```
brain/
  router.py       ← NEW: ChunkingProfile dataclass, signal detection, routing, salvage
  chunk.py        ← REFACTORED: pure chunking strategies only, no routing decisions
  ingest.py       ← LIGHT UPDATE: calls router, passes profile to chunker
  config.py       ← NEW CONSTANTS: semantic split threshold, stub min tokens
```

**Data flow:**

```
file_path
    ↓
extract_pages()              [chunk.py — unchanged]
    ↓
build_profile(path, pages)   [router.py]
    → ChunkingProfile
    ↓
route_and_chunk(profile, pages)  [router.py]
    → calls chunk_slides / chunk_semantic / chunk_structured / chunk_window
    ↓
salvage_pass(profile, pages, chunks)  [router.py]
    → appends stub chunks for garbled pages with extractable titles
    ↓
chunks with full metadata (strategy, is_stub, chunking_profile fields)
    ↓
ingest.py — quality filter → embed → index → log
```

---

## 3. ChunkingProfile

Defined in `brain/router.py`. Populated from signals before any chunking begins. Exposed as metadata on every chunk so routing decisions are auditable.

```python
@dataclass
class ChunkingProfile:
    strategy: str              # "slides" | "semantic" | "structured" | "window"
    avg_tokens_per_page: float
    has_structure: bool        # headings or PDF outline detected
    garbled_page_ratio: float  # fraction of pages that failed quality check
    file_extension: str
    page_count: int
```

---

## 4. Signal Detection and Routing

`build_profile(path, pages)` computes all signals. `route_and_chunk(profile, pages)` applies routing in this priority order:

| Priority | Condition | Strategy |
|---|---|---|
| 1 | Extension is `.docx` or `.md` | `structured` |
| 2 | Extension is `.pdf` AND avg tokens/page < `SLIDE_PAGE_TOKEN_THRESHOLD` | `slides` |
| 3 | Extension is `.pdf` AND outline/headings detected | `structured` |
| 4 | All other cases (dense PDF, `.txt`) | `semantic` |
| fallback | Semantic produces degenerate results | `window` |

Garbled page ratio is computed by running the existing `_is_quality_chunk` check against each page during profiling. If `garbled_page_ratio > 0`, the salvage pass runs after chunking regardless of strategy.

---

## 5. Chunking Strategies

### Slides
Group `SLIDE_PAGES_PER_CHUNK` consecutive pages. Detect topic transitions from the PDF outline — when a slide title changes between pages, prefer to start a new group at the transition rather than mid-section.

### Structured (DOCX, MD, outlined PDF)
Split on heading boundaries:
- **MD:** lines matching `^#{1,3}\s`
- **DOCX:** paragraphs with Heading1/Heading2 styles
- **PDF with outline:** use outline entries as section boundaries

Each heading + body = one chunk. If a section exceeds `CHUNK_SIZE_TOKENS`, apply overlapping window chunking within it.

### Semantic (dense PDF, TXT)
1. Split text into sentences using regex
2. Embed each sentence via the existing embed provider (batched)
3. Compute cosine similarity between consecutive sentence pairs
4. Split where similarity drops below `SEMANTIC_SPLIT_THRESHOLD` (default: 0.4)
5. Apply guardrails: minimum 100 tokens, maximum `CHUNK_SIZE_TOKENS`
6. Semantic splitting operates within each page — page boundaries are never crossed, preserving citation accuracy

If semantic chunking produces fewer than 2 splits (all sentences equally similar), fall back to `window` strategy.

### Window (fallback)
Existing 500-token overlapping window. Used when semantic produces degenerate results.

---

## 6. Salvage Pass

Runs after chunking on any document where `garbled_page_ratio > 0`.

For each page that failed the quality filter:
1. Extract the first non-empty line as a title candidate
2. Apply a lighter quality check: at least `MIN_STUB_TOKENS` (default: 3) meaningful words and no more than 20% single-character tokens
3. **Pass:** create a stub chunk with the title text, `is_stub: True`, same page/course/filename metadata as the original page
4. **Fail:** discard entirely as before

**Stub chunk example:**
A slide page containing a garbled Scrum cycle diagram with title "The Scrum Cycle" produces:
```python
{
  "id": "IS 6410_slides.pdf_p7_c0_stub",
  "text": "The Scrum Cycle",
  "is_stub": True,
  "strategy": "slides",
  "course": "IS 6410",
  "filename": "slides.pdf",
  "page": 7,
  ...
}
```

---

## 7. Chunk Metadata Changes

Every chunk gains two new fields:

| Field | Type | Description |
|---|---|---|
| `strategy` | str | Which chunking strategy produced this chunk |
| `is_stub` | bool | True if this is a salvage stub from a garbled page |

These fields are stored in ChromaDB metadata alongside the existing `course`, `filename`, `page`, `slide_title`, `chunk_index` fields.

---

## 8. CLAUDE.md Update

One additional instruction added:

> When a retrieved chunk has `is_stub: true` in its metadata, treat it as evidence that a topic exists in the source document only. Do not make factual claims sourced from stub chunks and do not cite them as factual sources.

The NLI citation verifier already rejects stub-sourced claims naturally (a title alone cannot entail a specific factual claim), but the explicit instruction prevents Claude from attempting to use stubs as evidence.

---

## 9. New Config Constants

| Constant | Default | Description |
|---|---|---|
| `SEMANTIC_SPLIT_THRESHOLD` | 0.4 | Cosine similarity drop below this triggers a semantic split |
| `MIN_STUB_TOKENS` | 3 | Minimum meaningful words for a title to become a stub chunk |

---

## 10. Files Changed

| File | Change |
|---|---|
| `brain/router.py` | **New.** ChunkingProfile, signal detection, routing, salvage pass |
| `brain/chunk.py` | **Refactored.** Remove `is_slide_deck`, add `chunk_semantic`, `chunk_structured`. `build_chunks_from_file` accepts a profile. |
| `brain/ingest.py` | **Light update.** Import router, call `build_profile` after extraction, pass profile to chunker |
| `brain/config.py` | **New constants.** `SEMANTIC_SPLIT_THRESHOLD`, `MIN_STUB_TOKENS` |
| `CLAUDE.md` | **One line added.** Stub chunk instruction |

---

## 11. Out of Scope

- Evaluation system (separate design — this is noted in future-ideas.md)
- Cross-page semantic chunking (would break citation page accuracy)
- AI-based document classification (signal-based router is sufficient and deterministic)
- Agentic/proposition-based chunking (too expensive for local ingestion)
