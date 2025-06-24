# retrieval/rerank.py
from sentence_transformers import CrossEncoder

ce_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)

def rerank(query, candidates):
    pairs = [(query, c["text"]) for c,_ in candidates]
    scores = ce_model.predict(pairs)
    return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
