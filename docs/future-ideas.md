# Future Ideas

A running list of features and extensions to build later.

---

## Query Access

### Local API Server
Wrap `brain/query.py` in a FastAPI server so the second brain can be queried from anywhere — Claude.ai, Cowork, a browser, or a mobile app. Would enable the full RAG pipeline (rewrite → retrieve → rerank → verify) outside of Claude Code CLI.

---

## Obsidian Integration

### Obsidian Plugin UI
A native Obsidian sidebar plugin that lets you ask questions and see cited answers without leaving the app. Would call the local API server above.

### Automatic Ingestion on File Save
Watch the `courses/` folder and automatically re-index new PDFs when they're added, instead of manually running `brain/ingest.py`.

---

## Study Tools

### Spaced Repetition / Flashcard Generation
Use the indexed content to auto-generate Anki-style flashcards from lecture slides. Could export to Anki's `.apkg` format or as Obsidian notes compatible with the Spaced Repetition plugin.

### Active Recall Quiz Mode
A mode where Claude asks *you* questions based on your course material instead of answering them — forces retrieval practice rather than passive review.

---

## Pipeline Improvements

### Multi-modal: Slide Images and Diagrams
Current pipeline extracts text only. Many slides contain diagrams, charts, and figures that carry meaning. Would require a vision model to caption or embed images alongside text chunks.

### Evaluation Script
A batch evaluation harness that runs a set of known questions against the pipeline and scores: retrieval precision, citation accuracy, and grounding ratio. Would make it easy to detect regressions when tuning config parameters.

### Cross-Course Synthesis
Explicitly prompt Claude to draw connections between concepts across different courses when answering — e.g. linking supply chain concepts from OSC 6660 to systems analysis in IS 6410.

---

## Infrastructure

### Multi-Vault / Multi-User Support
Extend the system to support multiple students sharing a knowledge base, or separate vaults per semester.
