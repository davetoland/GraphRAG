# embedding_store/embedder.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def embed(texts):
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
