# retrieval/dense.py
from embedding_store.embedder import embed

def dense_search(store, queries, top_k=10):
    vecs = embed(queries)
    return store.search(vecs, top_k)
