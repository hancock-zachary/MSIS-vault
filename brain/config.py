from pathlib import Path

VAULT_ROOT = Path(__file__).parent.parent
BRAIN_DIR = VAULT_ROOT / "brain"
CHROMA_DIR = BRAIN_DIR / "chroma"
BM25_PATH = BRAIN_DIR / "bm25.pkl"
INGESTION_LOG = BRAIN_DIR / "ingestion_log.json"

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVAL = 20      # per retrieval method per variant
TOP_K_RERANK = 8          # final chunks sent to Claude
RRF_K = 60                # RRF constant
RERANK_THRESHOLD = 0.0    # minimum cross-encoder score (0.0 = no filter)
ENTAILMENT_THRESHOLD = 0.3  # minimum score to consider a citation verified

EMBED_PROVIDER = "ollama"   # "ollama" | "openai"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

CHROMA_COLLECTION = "vault"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
