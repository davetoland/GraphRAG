# embedding_store/store.py
import faiss
import numpy as np

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # CPU index
        self.metadata = []

    def add(self, vectors: np.ndarray, metas: list):
        self.index.add(vectors)
        self.metadata.extend(metas)

    def search(self, query_vec, top_k=10):
        D, I = self.index.search(query_vec, top_k)
        results = []
        for j, idx in enumerate(I[0]):
            results.append((self.metadata[idx], float(D[0][j])))
        return results
