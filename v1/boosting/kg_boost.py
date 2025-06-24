# boosting/kg_boost.py
from config import BOOST_ALPHA, BOOST_BETA
import networkx as nx
from ner_kg.ner import extract_entities
from ner_kg.graph_builder import G

def boost_scores(candidates, query_text):
    qents = [e for e,_ in extract_entities(query_text)]
    boosted = []
    for (meta, base_score) in candidates:
        ent_list = meta["entities"]
        dists = []
        for q in qents:
            for e in ent_list:
                try:
                    dists.append(nx.shortest_path_length(G, q, e))
                except nx.NetworkXNoPath:
                    continue
        kg_score = 1.0 / (1 + min(dists or [100]))
        final   = BOOST_ALPHA * base_score + BOOST_BETA * kg_score
        boosted.append((meta, final))
    return sorted(boosted, key=lambda x: x[1], reverse=True)
