import faiss
import numpy as np
import pickle
from pathlib import Path
from config import NORMALIZE_EMBEDDINGS, EMBEDDING_DIM

class FaissStore:
    """HNSW-based FAISS store with metadata helpers."""

    def __init__(self, dim: int = EMBEDDING_DIM, hnsw_m: int = 32):
        # HNSW gives ~100× speed-up over brute-force while preserving recall.
        self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.metadata = []
        self.id_to_meta = {}
        self.dim = dim

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray, metas: list):
        if len(vectors) == 0:
            return  # nothing to add
        if NORMALIZE_EMBEDDINGS:
            faiss.normalize_L2(vectors)
        self.index.add(vectors.astype(np.float32))
        self.metadata.extend(metas)
        for m in metas:
            self.id_to_meta[m["id"]] = m

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        if NORMALIZE_EMBEDDINGS:
            faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec.astype(np.float32), top_k)
        results = []
        for j, idx in enumerate(I[0]):
            meta = self.metadata[idx]
            results.append((meta, float(D[0][j])))
        return results

    def get_meta(self, meta_id: str):
        return self.id_to_meta.get(meta_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, index_path: str, meta_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self.id_to_meta = {m["id"]: m for m in self.metadata}
