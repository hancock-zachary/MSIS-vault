from pathlib import Path

VAULT_ROOT = Path(__file__).parent.parent
COURSES_DIR = VAULT_ROOT / "courses"  # kept for reference
RAW_DIR = VAULT_ROOT / "raw"
BRAIN_DIR = VAULT_ROOT / "brain"
CHROMA_DIR = BRAIN_DIR / "chroma"
BM25_PATH = BRAIN_DIR / "bm25.pkl"
INGESTION_LOG = BRAIN_DIR / "ingestion_log.json"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
SLIDE_PAGE_TOKEN_THRESHOLD = 150  # avg tokens/page below this → treat as slide deck
SLIDE_PAGES_PER_CHUNK = 4         # consecutive pages grouped into one slide chunk
TOP_K_RETRIEVAL = 20      # per retrieval method per variant
TOP_K_RERANK = 8          # final chunks sent to Claude
RRF_K = 60                # RRF constant
RERANK_THRESHOLD = 0.0    # minimum cross-encoder score (0.0 = no filter)
ENTAILMENT_THRESHOLD = 0.5  # minimum entailment probability to consider a citation verified

EMBED_PROVIDER = "ollama"   # "ollama" | "openai"
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

NOTES_DIR = VAULT_ROOT / "notes"
GRAPH_TOP_K = 10           # maximum related documents per note (actual links may be fewer)
GRAPH_MIN_SIMILARITY = 0.6 # minimum avg similarity score to include a link at all

CHROMA_COLLECTION = "vault"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
