from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "indexes"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.idx"
FAISS_METADATA_PATH = INDEX_DIR / "faiss_meta.pkl"
KG_GRAPH_PATH = INDEX_DIR / "kg.graphml"

# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------
EMBEDDING_DIM = 768
NORMALIZE_EMBEDDINGS = True
EMBED_INSTRUCTION_DOC = "Represent the document for retrieval:"
EMBED_INSTRUCTION_QUERY = "Represent the query for retrieval:"

# ---------------------------------------------------------
# Chunking
# ---------------------------------------------------------
CHUNK_TOKEN_MIN = 120
CHUNK_TOKEN_MAX = 180
CHUNK_OVERLAP = 30

# ---------------------------------------------------------
# Retrieval & ranking
# ---------------------------------------------------------
DENSE_TOP_K = 10
SPARSE_TOP_K = 5
MERGE_ALPHA = 0.5
MMR_LAMBDA = 0.6
MMR_K = 10

# ---------------------------------------------------------
# KG boosting
# ---------------------------------------------------------
USE_KG_PREFILTER = False
BOOST_ALPHA = 0.8
BOOST_BETA = 0.2

# ---------------------------------------------------------
# Model names
# ---------------------------------------------------------
HF_LLM_NAME = "microsoft/Phi-3-mini-4k-instruct"
SENTENCE_TRANSFORMER = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER = "BAAI/bge-reranker-base"
