# Future Ideas

A running list of features and extensions, ordered by priority. Priorities reflect
the current goal: make chunking, embedding, and graphing as good as possible before
expanding coverage.

---

## Priority 1 — Retrieval quality fixes (high impact, low effort)

### Nomic Embedding Task Prefixes
`nomic-embed-text` is trained with task prefixes and underperforms without them, but
`src/embed.py` currently sends bare text. Fix: prepend `search_document: ` to chunk
text at indexing time and `search_query: ` to questions at query time (only for the
Ollama provider — OpenAI models don't use prefixes). This is closer to a bug fix than
a feature. Requires a full re-index since it changes every stored embedding.

### Evaluation Script
A batch evaluation harness (`src/eval.py`) that runs a set of known questions against
the pipeline and scores: retrieval precision (recall@k, MRR at both document and chunk
level), citation accuracy, and grounding ratio. Question set is auto-generated from
sampled indexed chunks (each question's gold answer = the chunk it came from), saved
to an editable JSON file. Makes every other tuning change below measurable and detects
regressions when config parameters change.

**Sequencing note:** capture a baseline score *before* the prefix-fix re-index wipes
the old embeddings, so the before/after comparison isn't lost.

### Contextual Chunk Enrichment
Prepend a small context header (`<course> · <filename> · <slide title>`) to each
chunk's text before embedding, so a chunk that says "the three phases are..." carries
its surrounding context into the vector. A lightweight version of Anthropic's
contextual retrieval. Store the clean text for display/citation; embed the enriched
text. Requires re-index — should land in the same re-index as the prefix fix.

---

## Priority 2 — Chunking and graph improvements (measured against the eval harness)

### Slide Chunk Boundary Improvements
`chunk_slides()` groups a fixed 4 pages per chunk with no overlap, so related slides
get split at arbitrary boundaries. Improvements to test against the eval harness:
1. Add a 1-page overlap between consecutive slide groups
2. When a PDF outline exists, split at section-title boundaries instead of fixed counts

### Smarter Wiki Excerpts
`graph.py` currently shows the first two chunks by page number, which is usually the
title slide and agenda. Instead pick the chunks closest to the document's embedding
centroid — the most representative content — so wiki pages and index previews actually
describe the document.

### Document Trust / Source Weighting
Different document types have different levels of authority. Lecture slides from a
professor are a primary source; assigned readings are secondary; student assignments
and coursework are the least authoritative and most likely to contain errors.

Implementation:
1. Tag each document at ingestion time with a `source_type` metadata field
   (e.g. `slides`, `reading`, `assignment`, `notes`), inferred from subfolder names
   under `raw/<Course>/`
2. Surface `source_type` in wiki page frontmatter and citations
3. Apply a configurable score multiplier during reranking — slides get a boost,
   assignments get a penalty (multipliers in `config.py`, tuned via the eval harness)

This becomes critical when assignments and coursework enter the index — an incorrect
answer you wrote on an exam should not be cited as a factual source.

### Config Threshold Tuning
Several config values are educated guesses: `SEMANTIC_SPLIT_THRESHOLD` (0.4),
`SLIDE_PAGES_PER_CHUNK` (4), `SLIDE_PAGE_TOKEN_THRESHOLD` (150),
`GRAPH_MIN_SIMILARITY` (0.72), `RRF_K`, `RERANK_THRESHOLD`. Sweep each against the
eval harness and keep the winners.

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

## Priority 4 — Access and tooling

### Local API Server
Wrap `src/query.py` in a FastAPI server so the second brain can be queried from
anywhere — Claude.ai, Cowork, a browser, or a mobile app. Would enable the full RAG
pipeline (rewrite → retrieve → rerank → verify) outside of Claude Code CLI.

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

## Priority 6 — Infrastructure

### Multi-Vault / Multi-User Support
Extend the system to support multiple students sharing a knowledge base, or separate
vaults per semester.

---

## Done

- ~~Slide-Aware Chunking Strategy~~ ✅
- ~~Real Entailment Verification (NLI model)~~ ✅
