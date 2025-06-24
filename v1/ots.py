# one time setup
from config import FAISS_DIM
from pathlib import Path
from ingestion.reader import read_file
from ingestion.chunker import chunk_elements
from ner_kg.ner import extract_entities
from ner_kg.graph_builder import add_chunk_entities, save_graph
from embedding_store.embedder import embed
from embedding_store.store import FaissStore

store = FaissStore(dim=FAISS_DIM)

for path in Path("docs").glob("*.*"):
    elems  = read_file(path)
    chunks = chunk_elements(elems)
    texts  = [c["text"] for c in chunks]
    vecs   = embed(texts)
    metas  = []
    for i, c in enumerate(chunks):
        ents = extract_entities(c["text"])
        add_chunk_entities(f"{path.name}_{i}", ents)
        metas.append({
            "id":       f"{path.name}_{i}",
            "entities": [e for e,_ in ents],
            **c["meta"]
        })
    store.add(vecs, metas)

save_graph(str(Path("indexes")/"kg.graphml"))
