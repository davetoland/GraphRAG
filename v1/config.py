# config.py
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
MODEL_DIR        = BASE_DIR / "models"
INDEX_DIR        = BASE_DIR / "indexes"

# One‐time: place your GGML model here
LLAMA_MODEL_PATH = MODEL_DIR / "ggml-model.bin"

# ─── FAISS (CPU) ─────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = INDEX_DIR / "faiss.idx"
FAISS_DIM        = 384      # match your sentence-transformer output size

# ─── KG-BOOST WEIGHTS ────────────────────────────────────────────────────────
BOOST_ALPHA      = 0.8      # semantic (cross-encoder) weight
BOOST_BETA       = 0.2      # KG proximity weight

# ─── llama-cpp-python (local LLM) ────────────────────────────────────────────
# If you built with CUDA/cuBLAS support set to True
LLAMA_USE_GPU    = True

LLAMA_KWARGS     = {
    "max_tokens":  64,
    "temperature": 0.7,
    "top_p":       0.9,
}
