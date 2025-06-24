# ner_kg/graph_builder.py
import networkx as nx
from typing import List, Tuple

G = nx.Graph()

def add_chunk_entities(chunk_id: str, entities: List[Tuple[str,str]]):
    for ent, label in entities:
        G.add_node(ent, label=label)
        G.add_edge(chunk_id, ent)

def save_graph(path: str):
    nx.write_graphml(G, path)

def load_graph(path: str):
    global G
    G = nx.read_graphml(path)
