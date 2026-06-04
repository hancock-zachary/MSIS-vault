# Future Ideas

A running list of features and extensions to build later.

---

## Query Access

### Local API Server
Wrap `src/query.py` in a FastAPI server so the second brain can be queried from anywhere — Claude.ai, Cowork, a browser, or a mobile app. Would enable the full RAG pipeline (rewrite → retrieve → rerank → verify) outside of Claude Code CLI.

---

## Obsidian Integration

### Obsidian Plugin UI
A native Obsidian sidebar plugin that lets you ask questions and see cited answers without leaving the app. Would call the local API server above.

### Automatic Ingestion on File Save
Watch the `courses/` folder and automatically re-index new PDFs when they're added, instead of manually running `src/ingest.py`.

---

## Study Tools

### Spaced Repetition / Flashcard Generation
Use the indexed content to auto-generate Anki-style flashcards from lecture slides. Could export to Anki's `.apkg` format or as Obsidian notes compatible with the Spaced Repetition plugin.

### Active Recall Quiz Mode
A mode where Claude asks *you* questions based on your course material instead of answering them — forces retrieval practice rather than passive review.

---

## Pipeline Improvements

### Document Trust / Source Weighting
Different document types have different levels of authority. Lecture slides from a professor are a primary source; assigned readings are secondary; student assignments and coursework are the least authoritative and most likely to contain errors or incomplete reasoning. The pipeline currently treats all documents equally during retrieval and reranking.

Implementing trust tiers would work by:
1. Tagging each document at ingestion time with a `source_type` metadata field (e.g. `slides`, `reading`, `assignment`, `notes`)
2. Applying a score multiplier during reranking — slides get a boost, assignments get a penalty
3. Making the multipliers configurable in `config.py` so they can be tuned over time

This becomes critical when assignments and coursework enter the index — an incorrect answer you wrote on an exam should not be cited as a factual source.

### ~~Slide-Aware Chunking Strategy~~ ✅ Done

### Multi-modal: Slide Images and Diagrams
Current pipeline extracts text only. Many slides contain diagrams, charts, and figures that carry meaning. Would require a vision model to caption or embed images alongside text chunks.

### Audio/Visual Ingestion
Lecture recordings, video walkthroughs, and recorded office hours contain content that never appears in slides. Ingesting these would make the second brain significantly more complete, especially for courses where the professor covers material verbally that isn't in the slides.

The pipeline would be:
1. **Audio extraction** — pull audio track from video files (MP4, MOV, etc.) using `ffmpeg`
2. **Transcription** — convert speech to text using a local Whisper model (`openai-whisper` or `faster-whisper`) — runs on CPU, no API cost
3. **Chunking** — split transcripts by time window (e.g. 2-minute segments) rather than token count, preserving temporal context
4. **Indexing** — same ChromaDB + BM25 pipeline as text documents, with metadata including timestamp so citations can reference the video at a specific time (e.g. `[source: lecture3.mp4, 14:32]`)

The biggest wins would be: capturing worked examples explained verbally, Q&A sessions, and any content a professor adds beyond the slide deck.

### ~~Real Entailment Verification (NLI model)~~ ✅ Done

### Evaluation Script
A batch evaluation harness that runs a set of known questions against the pipeline and scores: retrieval precision, citation accuracy, and grounding ratio. Would make it easy to detect regressions when tuning config parameters.

### Cross-Course Synthesis
Explicitly prompt Claude to draw connections between concepts across different courses when answering — e.g. linking supply chain concepts from OSC 6660 to systems analysis in IS 6410.

---

## Infrastructure

### Multi-Vault / Multi-User Support
Extend the system to support multiple students sharing a knowledge base, or separate vaults per semester.
