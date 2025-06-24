# prefilter/kg_filter.py
from ner_kg.ner import extract_entities

def filter_by_kg(candidates, query_text):
    qents = {e for e,_ in extract_entities(query_text)}
    return [c for c in candidates if qents & set(c[0]["entities"])]
