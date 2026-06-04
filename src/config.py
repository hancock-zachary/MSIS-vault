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
BOILERPLATE_PAGE_RATIO = 0.4      # line appearing on >40% of pages is boilerplate
BOILERPLATE_MIN_PAGES = 4         # min pages needed before boilerplate detection runs
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

WIKI_DIR = VAULT_ROOT / "wiki"
GRAPH_TOP_K = 10           # candidate pool size per document before mutual filtering
GRAPH_MIN_SIMILARITY = 0.72 # minimum avg similarity — filters weak vocabulary overlap

CHROMA_COLLECTION = "vault"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
