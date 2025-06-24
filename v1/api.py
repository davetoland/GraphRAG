# api.py
from fastapi import FastAPI
from ingestion.reader import read_file
from ingestion.chunker import chunk_elements
from ner_kg.ner import extract_entities
from ner_kg.graph_builder import add_chunk_entities, save_graph
from embedding_store.embedder import embed
from embedding_store.store import FaissStore
from retrieval.bm25 import BM25Index
from retrieval.dense import dense_search
from retrieval.rerank import rerank
from expansion.expander import expand_query
from prefilter.kg_filter import filter_by_kg
from boosting.kg_boost import boost_scores

app = FastAPI()

# (Define endpoints to ingest and to query, wiring the pipeline as needed)
