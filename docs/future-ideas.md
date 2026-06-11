# Future Ideas

A running list of features and extensions, ordered by priority. Priorities reflect
the current goal: make chunking, embedding, and graphing as good as possible before
expanding coverage.

---

## Priority 1 — Correctness and retrieval-quality fixes (high impact, low effort)

_All complete — see Done section._

---

## Priority 2 — Chunking, retrieval, and graph improvements (measured against the eval harness)

### Smarter Wiki Excerpts
`graph.py` currently shows the first two chunks by page number, which is usually the
title slide and agenda. Instead pick the chunks closest to the document's embedding
centroid — the most representative content — so wiki pages and index previews actually
describe the document.

### Diversity in Final Selection (MMR)
The cross-encoder reranker can return 8 near-duplicate chunks from the same page.
Maximal-marginal-relevance selection trades a little relevance for broader coverage
of the question. Tune the relevance/diversity balance against the eval harness.

### Config Threshold Tuning
Several config values are educated guesses: `SEMANTIC_SPLIT_THRESHOLD` (0.4),
`SLIDE_PAGES_PER_CHUNK` (4), `SLIDE_PAGE_TOKEN_THRESHOLD` (150),
`GRAPH_MIN_SIMILARITY` (0.72), `RRF_K`, `RERANK_THRESHOLD`. Sweep each against the
eval harness and keep the winners.

### Answer-Level Evaluation (Citation Accuracy + Grounding Ratio)
The eval harness (`src/eval.py`) scores retrieval only. Extend it to optionally
generate full answers via Claude for each eval question and score citation accuracy
(does each cited filename/page actually match the gold source?) and grounding ratio
(what fraction of claims pass NLI entailment verification?). Costly per run — keep it
a separate opt-in mode from the fast retrieval eval.

### Cross-Course Graph Links + Synthesis
Label cross-course links distinctly on wiki pages (they're the most interesting kind),
and explicitly prompt Claude to draw connections between concepts across courses when
answering — e.g. linking supply chain concepts from OSC 6660 to systems analysis in
IS 6410.

---

## Priority 3 — Coverage expansion (more content into the index)

### Multi-modal: Slide Images and Diagrams
Current pipeline extracts text only. Many slides contain diagrams, charts, and figures
that carry meaning (today these become `is_stub` salvage chunks at best). Would require
a vision model to caption or embed images alongside text chunks.

### Audio/Visual Ingestion
Lecture recordings, video walkthroughs, and recorded office hours contain content that
never appears in slides. The pipeline would be:
1. **Audio extraction** — pull audio track from video files (MP4, MOV, etc.) using `ffmpeg`
2. **Transcription** — convert speech to text using a local Whisper model
   (`openai-whisper` or `faster-whisper`) — runs on CPU, no API cost
3. **Chunking** — split transcripts by time window (e.g. 2-minute segments) rather than
   token count, preserving temporal context
4. **Indexing** — same ChromaDB + BM25 pipeline as text documents, with metadata
   including timestamp so citations can reference the video at a specific time
   (e.g. `[source: lecture3.mp4, 14:32]`)

The biggest wins: worked examples explained verbally, Q&A sessions, and content a
professor adds beyond the slide deck.

---

## Priority 4 — Access, tooling, and performance

### Local API Server
Wrap `src/query.py` in a FastAPI server so the second brain can be queried from
anywhere — Claude.ai, Cowork, a browser, or a mobile app. Would enable the full RAG
pipeline (rewrite → retrieve → rerank → verify) outside of Claude Code CLI.

### Query Rewrite Caching
Every query shells out to `claude -p` for rewriting (up to 30s, costs tokens). Cache
rewrites by question hash, and/or fall back to a local model, to cut latency and cost
for repeated or similar questions.

### Obsidian Plugin UI
A native Obsidian sidebar plugin that lets you ask questions and see cited answers
without leaving the app. Would call the local API server above.

### Automatic Ingestion on File Save
Watch the `raw/` folder and automatically re-index new PDFs when they're added,
instead of manually running `src/ingest.py`.

---

## Priority 5 — Study tools

### Spaced Repetition / Flashcard Generation
Use the indexed content to auto-generate Anki-style flashcards from lecture slides.
Could export to Anki's `.apkg` format or as Obsidian notes compatible with the
Spaced Repetition plugin.

### Active Recall Quiz Mode
A mode where Claude asks *you* questions based on your course material instead of
answering them — forces retrieval practice rather than passive review.

---

## Priority 6 — Infrastructure and engineering hygiene

### Multi-Vault / Multi-User Support
Extend the system to support multiple students sharing a knowledge base, or separate
vaults per semester.

---

## Done

- ~~Slide-Aware Chunking Strategy~~ ✅
- ~~Real Entailment Verification (NLI model)~~ ✅
- ~~Nomic Embedding Task Prefixes~~ ✅ (code done — full re-index still required so
  stored document embeddings match the new prefixed format)
- ~~Evaluation Script~~ ✅ (`src/eval.py` — retrieval metrics: doc/passage recall@k
  and MRR at retrieval + rerank stages, auto-generated question set, baseline
  comparison; answer-level metrics moved to a separate P2 item)
- ~~Document Trust / Source Weighting~~ ✅ (`source_type` inferred from subfolders
  under `raw/<Course>/` at ingest, surfaced in wiki frontmatter and query context
  headers, and applied as an additive rerank bonus via `SOURCE_TYPE_RERANK_BONUS` —
  additive, not the multiplier originally sketched, because cross-encoder logits can
  be negative; bonuses inert until re-index stamps the metadata)
- ~~Neighbor Chunk Expansion (Small-to-Big)~~ ✅ (reranked winners absorb page-adjacent
  chunk text via `NEIGHBOR_PAGE_WINDOW`; each neighbor claimed once so context never
  duplicates; page ranges widen so citations to merged pages still verify)
- ~~Course-Scoped Retrieval~~ ✅ (`--course` flag on query.py plus auto-detection when
  the question mentions exactly one known course; filters both Chroma and BM25;
  multi-course questions stay unscoped for cross-course synthesis)
- ~~Slide Chunk Boundary Improvements~~ ✅ (1-page overlap between slide groups via
  `SLIDE_PAGE_OVERLAP`; outlined decks split at section-title boundaries with the
  section title carried onto every window; validate against the eval baseline after
  re-indexing)
- ~~Single Source of Truth for BM25~~ ✅ (BM25 pickle is now a derived cache rebuilt
  from ChromaDB at the end of every ingest run — crashes and purges can no longer
  leave the two stores out of sync)
- ~~Stub Down-weighting in Retrieval~~ ✅ (RRF scores of stubs scaled by
  `STUB_RRF_MULTIPLIER`; reranker always orders stubs after content chunks)
- ~~Better BM25 Tokenization~~ ✅ (punctuation-splitting + conservative stopword list,
  applied identically at index and query time; takes effect on BM25 rebuild)
- ~~Contextual Chunk Enrichment~~ ✅ (course · document · slide-title header prepended
  at embedding time only; stored chunk text stays clean; takes effect on re-index)
- ~~Page Ranges for Slide Chunks~~ ✅ (slide chunks store `page_start`/`page_end`,
  verify.py matches citations within the range, query.py shows "pages X-Y"; chunks
  indexed before the re-index lack range metadata and fall back to exact-page match)
