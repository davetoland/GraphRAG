import spacy
import torch
import networkx as nx
from typing import List, Dict, Tuple
from config import (
    HF_LLM_NAME, SENTENCE_TRANSFORMER, CROSS_ENCODER,
    CHUNK_TOKEN_MIN, CHUNK_TOKEN_MAX, CHUNK_OVERLAP,
    USE_KG_PREFILTER, BOOST_ALPHA, BOOST_BETA, NORMALIZE_EMBEDDINGS,
    EMBED_INSTRUCTION_DOC, EMBED_INSTRUCTION_QUERY
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from unstructured.partition.auto import partition
from pathlib import Path
import numpy as np

# ------------------------------------------------------------------
# spaCy (transformer model preferred)
# ------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------------
# Torch device helper
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding / reranker
embed_model = SentenceTransformer(SENTENCE_TRANSFORMER, device=DEVICE)
reranker_model = CrossEncoder(CROSS_ENCODER, device=DEVICE)

# Query-expansion LLM
tok = AutoTokenizer.from_pretrained(HF_LLM_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(
    HF_LLM_NAME,
    device_map="cpu",
    torch_dtype=torch.float16   # or bfloat16
)
llm_gen = pipeline("text-generation", model=llm_model, tokenizer=tok, do_sample=False)

# Knowledge graph
G = nx.Graph()

# ------------------------------------------------------------------
# Token length util (tokeniser from HF LLM)
# ------------------------------------------------------------------
_tok = AutoTokenizer.from_pretrained("gpt2")  # cheap BPE for length estimate

def _token_len(text: str) -> int:
    return len(_tok.tokenize(text))

# ------------------------------------------------------------------
# Embedding helpers
# ------------------------------------------------------------------

def embed(texts: List[str], *, is_query: bool = False) -> np.ndarray:
    """Encode list of texts with BGE instruction prefixes."""
    prefix = EMBED_INSTRUCTION_QUERY if is_query else EMBED_INSTRUCTION_DOC
    prepared = [f"{prefix} {t}" for t in texts]
    vecs = embed_model.encode(prepared, show_progress_bar=False, convert_to_numpy=True)
    if NORMALIZE_EMBEDDINGS:
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-6, None)
    return vecs

# ------------------------------------------------------------------
# Chunking with overlap
# ------------------------------------------------------------------

def chunk_elements(path: Path) -> List[Dict]:
    elements = partition(str(path))
    chunks, buffer, buffer_meta = [], [], []
    token_count, chunk_idx = 0, 0

    def _flush(buf, meta, idx):
        combined = " ".join(buf)
        metadata = {
            "id": f"{path.name}:{idx}",
            "source": path.name,
            "block_idx": idx,
            "element_types": meta,
        }
        return {"text": combined, "meta": metadata}

    for el in elements:
        text = (el.text or "").strip()
        if not text:
            continue
        tlen = _token_len(text)
        # Start new chunk if (a) max would be breached and (b) we’re past min
        if token_count + tlen > CHUNK_TOKEN_MAX and token_count >= CHUNK_TOKEN_MIN:
            chunks.append(_flush(buffer, buffer_meta, chunk_idx))
            chunk_idx += 1
            # overlap
            buffer = buffer[-CHUNK_OVERLAP:]
            buffer_meta = buffer_meta[-CHUNK_OVERLAP:]
            token_count = sum(_token_len(x) for x in buffer)
        buffer.append(text)
        buffer_meta.append(getattr(el, "category", "Unknown"))
        token_count += tlen

    if buffer:
        chunks.append(_flush(buffer, buffer_meta, chunk_idx))
    return chunks

# ------------------------------------------------------------------
# Entity extraction + KG
# ------------------------------------------------------------------

def extract_entities(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

CHUNK_LABELS = {"PERSON", "ORG", "GPE", "DATE", "TIME", "EMAIL", "EVENT"}

def add_chunk_entities(chunk_id: str, entities: List[Tuple[str, str]]):
    G.add_node(chunk_id, type="chunk")
    for ent, label in entities:
        if label not in CHUNK_LABELS or len(ent) < 3 or ent.isdigit():
            continue
        G.add_node(ent, type="entity", label=label)
        G.add_edge(chunk_id, ent, relation="mentions")


def save_graph(path: str):
    nx.write_graphml(G, path)


def load_graph(path: str):
    global G
    G = nx.read_graphml(path)

# ------------------------------------------------------------------
# Dense retrieval (RRF aggregation across paraphrases)
# ------------------------------------------------------------------

def dense_search(store, queries: List[str], top_k: int = 10, rrf_k: int = 60):
    vecs = embed(queries, is_query=True)
    combined = {}
    for vec in vecs:
        res = store.search(vec[np.newaxis, :], top_k)
        for rank, (meta, _) in enumerate(res):
            entry = combined.setdefault(meta["id"], {"meta": meta, "score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank)
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [(r["meta"], r["score"]) for r in ranked[:top_k]]

# ------------------------------------------------------------------
# Reranking (cross-encoder)
# ------------------------------------------------------------------

def rerank(query: str, candidates: List[Tuple[Dict, float]]):
    pairs = [(query, m["text"]) for m, _ in candidates]
    scores = reranker_model.predict(pairs)
    rescored = [(m, float(s)) for (m, _), s in zip(candidates, scores)]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored

# ------------------------------------------------------------------
# KG soft-boost
# ------------------------------------------------------------------

def boost_scores(candidates: List[Tuple[Dict, float]], query_text: str):
    qents = [e for e, _ in extract_entities(query_text)]
    boosted = []
    for meta, base in candidates:
        ent_list = meta.get("entities", [])
        dists = []
        for qe in qents:
            for ce in ent_list:
                try:
                    dists.append(nx.shortest_path_length(G, qe, ce))
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
        kg_score = 1.0 / (1 + min(dists or [100]))
        boosted.append((meta, BOOST_ALPHA * base + BOOST_BETA * kg_score))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted

# ------------------------------------------------------------------
# Query expansion (3 paraphrases, deterministic)
# ------------------------------------------------------------------

def expand_query(text: str) -> List[str]:
    ents = [e for e, _ in extract_entities(text)]
    related = []
    for e in ents:
        related.extend(list(G.neighbors(e)))
    related = related[:5]

    prompt = (
        "Generate three diverse paraphrases of the following query.\n\n"
        f"Query: {text}\n"
    )
    if related:
        prompt += f"\nOptional related terms: {', '.join(related)}"

    outputs = llm_gen(prompt, max_new_tokens=64, num_return_sequences=3)
    return [o["generated_text"].split("Query:")[-1].strip() for o in outputs]
