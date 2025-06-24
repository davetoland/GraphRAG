from pathlib import Path
from grag import chunk_elements, extract_entities, add_chunk_entities, save_graph, embed
from faiss_store import FaissStore
from config import EMBEDDING_DIM, FAISS_INDEX_PATH, FAISS_METADATA_PATH, KG_GRAPH_PATH

# Build FAISS store and KG in one pass
store = FaissStore(dim=EMBEDDING_DIM)

for path in Path("docs").glob("*.*"):
    # Chunk and embed
    chunks = chunk_elements(path)
    texts = [c["text"] for c in chunks]
    vecs = embed(texts)

    # Build metadata and KG
    metas = []
    for c in chunks:
        chunk_id = c["meta"]["id"]
        ents = extract_entities(c["text"])
        add_chunk_entities(chunk_id, ents)
        metas.append({
            **c["meta"],
            "entities": [e for e, _ in ents]
        })

    store.add(vecs, metas)

# Persist index and graph
store.save(str(FAISS_INDEX_PATH), str(FAISS_METADATA_PATH))
save_graph(str(KG_GRAPH_PATH))
