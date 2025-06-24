# retrieval/bm25.py
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, docs):
        tokenized = [doc.split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)
        self.docs = docs

    def query(self, q, top_k=10):
        scores = self.bm25.get_scores(q.split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.docs[i], scores[i]) for i in top_idx]
