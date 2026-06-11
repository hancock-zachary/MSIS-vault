from pathlib import Path

VAULT_ROOT = Path(__file__).parent.parent
COURSES_DIR = VAULT_ROOT / "courses"  # kept for reference
RAW_DIR = VAULT_ROOT / "raw"
BRAIN_DIR = VAULT_ROOT / "src"
CHROMA_DIR = BRAIN_DIR / "chroma"
BM25_PATH = BRAIN_DIR / "bm25.pkl"
INGESTION_LOG = BRAIN_DIR / "ingestion_log.json"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
SEMANTIC_SPLIT_THRESHOLD = 0.4  # cosine similarity drop below this triggers a semantic split
MIN_STUB_TOKENS = 3              # minimum meaningful words for a garbled-page title to become a stub
SLIDE_PAGE_TOKEN_THRESHOLD = 150  # avg tokens/page below this → treat as slide deck
SLIDE_PAGES_PER_CHUNK = 4         # consecutive pages grouped into one slide chunk
SLIDE_PAGE_OVERLAP = 1            # pages shared between consecutive slide groups
BOILERPLATE_PAGE_RATIO = 0.4      # line appearing on >40% of pages is boilerplate
BOILERPLATE_MIN_PAGES = 4         # min pages needed before boilerplate detection runs
TOP_K_RETRIEVAL = 20      # per retrieval method per variant
TOP_K_RERANK = 8          # final chunks sent to Claude
RRF_K = 60                # RRF constant
RERANK_THRESHOLD = 0.0    # minimum cross-encoder score (0.0 = no filter)
STUB_RRF_MULTIPLIER = 0.3 # stub chunks' RRF scores are scaled down by this factor
NEIGHBOR_PAGE_WINDOW = 1  # pages on each side merged into reranked winners (small-to-big)

# Document trust: subfolder-name keyword (under raw/<Course>/) → canonical source type
SOURCE_TYPE_KEYWORDS = {
    "slide": "slides", "lecture": "slides", "deck": "slides",
    "reading": "reading", "textbook": "reading", "article": "reading",
    "transcript": "transcript",
    "assignment": "assignment", "homework": "assignment", "exam": "assignment",
    "quiz": "assignment", "coursework": "assignment",
    "note": "notes",
}
# Additive rerank bonus per source type. Additive (not multiplicative)
# because cross-encoder logits can be negative — a multiplier would turn
# a penalty into a boost on negative scores. Tune against the eval harness.
SOURCE_TYPE_RERANK_BONUS = {
    "slides": 0.5,       # professor's primary material
    "transcript": 0.25,  # professor's spoken material
    "reading": 0.0,      # assigned secondary sources
    "notes": -0.25,      # student-authored notes
    "assignment": -1.0,  # student work — least authoritative, may contain errors
    "unknown": 0.0,
}
ENTAILMENT_THRESHOLD = 0.5  # minimum entailment probability to consider a citation verified

EMBED_PROVIDER = "ollama"   # "ollama" | "openai"
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

EVAL_DIR = VAULT_ROOT / "eval"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"
EVAL_BASELINE_PATH = EVAL_DIR / "baseline.json"
EVAL_RESULTS_DIR = EVAL_DIR / "results"
EVAL_QUESTION_COUNT = 50    # default questions to auto-generate
EVAL_SNIPPET_CHARS = 120    # gold snippet length for passage-level matching
EVAL_MIN_CHUNK_CHARS = 200  # chunks shorter than this are skipped during generation
EVAL_SAMPLE_SEED = 42       # fixed seed so question generation is reproducible

WIKI_DIR = VAULT_ROOT / "wiki"
GRAPH_TOP_K = 10           # candidate pool size per document before mutual filtering
GRAPH_MIN_SIMILARITY = 0.72 # minimum avg similarity — filters weak vocabulary overlap

CHROMA_COLLECTION = "vault"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
