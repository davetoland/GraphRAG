# import io
# import traceback
# import networkx as nx
# import matplotlib.pyplot as plt
# from pathlib import Path
# from fastapi.responses import StreamingResponse, JSONResponse
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from config import (
#     FAISS_INDEX_PATH, FAISS_METADATA_PATH, KG_GRAPH_PATH,
#     DENSE_TOP_K, SPARSE_TOP_K, MERGE_ALPHA
# )
# from faiss_store import FaissStore
# from bm25_index import BM25Index
# import grag
# import numpy as np
#
# from docx import Document
#
# app = FastAPI()
#
# # ------------------------------------------------------------------
# # Stores (will be filled on startup)
# # ------------------------------------------------------------------
# faiss_store = FaissStore(dim=grag.embed_model.get_sentence_embedding_dimension())
# bm25_index = None
# all_chunk_texts = []
#
# # ------------------------------------------------------------------
# # Startup: load persisted artefacts
# # ------------------------------------------------------------------
#
# def startup_event():
#     global bm25_index, all_chunk_texts
#     if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
#         faiss_store.load(str(FAISS_INDEX_PATH), str(FAISS_METADATA_PATH))
#         all_chunk_texts = [m["text"] for m in faiss_store.metadata]
#         bm25_index = BM25Index(all_chunk_texts)
#     if Path(KG_GRAPH_PATH).exists():
#         grag.load_graph(str(KG_GRAPH_PATH))
#
#
# # app.on_event("startup")(startup_event)
#
# # ------------------------------------------------------------------
# # Helpers
# # ------------------------------------------------------------------
#
# def _min_max(x):
#     x = np.array(x, dtype=float)
#     return np.zeros_like(x) if x.max() == x.min() else (x - x.min()) / (x.max() - x.min())
#
# # ------------------------------------------------------------------
# # Ingestion endpoint
# # ------------------------------------------------------------------
#
# @app.post("/ingest")
# async def ingest(file: UploadFile = File(...)):
#     docs_dir = Path(__file__).parent / "docs"
#     docs_dir.mkdir(exist_ok=True)
#     path = docs_dir / file.filename
#     try:
#         path.write_bytes(await file.read())
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
#
#     # Chunk & embed
#     chunks = grag.chunk_elements(path)
#     texts = [c["text"] for c in chunks]
#     vecs = grag.embed(texts, is_query=False)
#
#     metas = []
#     for c in chunks:
#         cid = c["meta"]["id"]
#         ents = grag.extract_entities(c["text"])
#         grag.add_chunk_entities(cid, ents)
#         metas.append({**c["meta"], "entities": [e for e, _ in ents]})
#
#     faiss_store.add(vecs, metas)
#     faiss_store.save(str(FAISS_INDEX_PATH), str(FAISS_METADATA_PATH))
#
#     global bm25_index, all_chunk_texts
#     all_chunk_texts.extend(texts)
#     bm25_index = BM25Index(all_chunk_texts)
#
#     grag.save_graph(str(KG_GRAPH_PATH))
#     return {"ingested_chunks": len(chunks)}
#
# @app.post("/ingest2")
# async def ingest(file: UploadFile = File(...)):
#     docs_dir = Path(__file__).parent / "docs"
#     docs_dir.mkdir(exist_ok=True)
#     path = docs_dir / file.filename
#     try:
#         path.write_bytes(await file.read())
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
#
#     document = Document(path)
#     non_empty = [p for p in document.paragraphs if p.text.strip()]
#
#     grph = nx.Graph()
#     grph.add_node(file.filename)
#     last_entity = file.filename
#     for ne in non_empty:
#         if ne.paragraph_format.element.style is not None:
#             grph.add_node(ne.text, type="entity")
#             grph.add_edge(file.filename, ne.text)
#             last_entity = ne.text
#         else:
#             grph.add_node(ne.text, type="chunk")
#             grph.add_edge(last_entity, ne.text)
#
#
#     grag.G = grph
#     return await get_graph()
#
#
#     # Chunk & embed
#     chunks = grag.chunk_elements(path)
#     texts = [c["text"] for c in chunks]
#     vecs = grag.embed(texts, is_query=False)
#
#     metas = []
#     for c in chunks:
#         cid = c["meta"]["id"]
#         ents = grag.extract_entities(c["text"])
#         grag.add_chunk_entities(cid, ents)
#         metas.append({**c["meta"], "entities": [e for e, _ in ents]})
#
#     faiss_store.add(vecs, metas)
#     faiss_store.save(str(FAISS_INDEX_PATH), str(FAISS_METADATA_PATH))
#
#     global bm25_index, all_chunk_texts
#     all_chunk_texts.extend(texts)
#     bm25_index = BM25Index(all_chunk_texts)
#
#     grag.save_graph(str(KG_GRAPH_PATH))
#     return {"ingested_chunks": len(chunks)}
#
# # ------------------------------------------------------------------
# # Query endpoint
# # ------------------------------------------------------------------
#
# @app.post("/query")
# async def query(q: str):
#     if bm25_index is None:
#         raise HTTPException(status_code=400, detail="No documents ingested yet.")
#
#     expanded = grag.expand_query(q)
#
#     dense = grag.dense_search(faiss_store, expanded, top_k=DENSE_TOP_K)
#     sparse = bm25_index.query(q, top_k=SPARSE_TOP_K)
#
#     dense_norm = _min_max([s for _, s in dense]).tolist()
#     sparse_norm = _min_max([s for _, s in sparse]).tolist()
#
#     merged = {}
#     # dense
#     for (meta, _), s in zip(dense, dense_norm):
#         merged[meta["id"]] = {"meta": meta, "dense": s, "sparse": 0.0}
#     # sparse
#     for (text, _), s in zip(sparse, sparse_norm):
#         for m in faiss_store.metadata:
#             if m["text"] == text:
#                 entry = merged.setdefault(m["id"], {"meta": m, "dense": 0.0, "sparse": 0.0})
#                 entry["sparse"] = s
#
#     candidates = [(v["meta"], MERGE_ALPHA * v["dense"] + (1 - MERGE_ALPHA) * v["sparse"]) for v in merged.values()]
#
#     try:
#         reranked = grag.rerank(q, candidates)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Rerank error: {e}\n{traceback.format_exc()}")
#
#     boosted = grag.boost_scores(reranked, q)
#
#     return {
#         "query": q,
#         "expanded": expanded,
#         "results": [
#             {
#                 "id": m["id"],
#                 "text": m["text"],
#                 "source": m.get("source"),
#                 "block_idx": m.get("block_idx"),
#                 "entities": m.get("entities", []),
#                 "score": float(s),
#             }
#             for m, s in boosted[:DENSE_TOP_K]
#         ],
#     }
#
# # ------------------------------------------------------------------
# # KG visualisation
# # ------------------------------------------------------------------
#
# @app.get("/graph", response_class=StreamingResponse)
# async def get_graph():
#     if grag.G.number_of_nodes() == 0:
#         raise HTTPException(status_code=404, detail="Knowledge graph is empty")
#
#     fig, ax = plt.subplots(figsize=(18, 16))
#     pos = nx.spring_layout(grag.G, seed=42)
#     colors = ["lightgreen" if grag.G.nodes[n].get("type") == "chunk" else "skyblue" for n in grag.G]
#     nx.draw_networkx(grag.G, pos, node_color=colors, with_labels=True, node_size=300, font_size=7)
#     ax.axis("off")
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", bbox_inches="tight")
#     plt.close(fig)
#     buf.seek(0)
#     return StreamingResponse(buf, media_type="image/png", headers={"Cache-Control": "no-store"})
#
#
# @app.get("/graph/text")
# async def graph_text():
#     nodes = [{"id": n, **attrs} for n, attrs in grag.G.nodes(data=True)]
#     edges = [
#         {"source": u, "target": v, **grag.G.edges[u, v]}
#         for u, v in grag.G.edges()
#     ]
#     return JSONResponse({"nodes": nodes, "edges": edges})
