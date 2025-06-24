# ingestion/chunker.py
import spacy
spacy.require_cpu()     # ensure CPU usage; remove if you want GPU
nlp = spacy.load("en_core_web_sm")

def chunk_elements(elements, max_tokens: int = 500):
    """
    elements: list of text blocks from unstructured.partition
    returns: list of {"text": str, "meta": {...}}
    """
    chunks, current, token_count = [], [], 0
    for idx, block in enumerate(elements):
        doc = nlp(block)
        for sent in doc.sents:
            stoks = len(sent)
            if token_count + stoks > max_tokens:
                chunks.append({
                    "text": " ".join(current),
                    "meta": {"last_block_idx": idx}
                })
                current, token_count = [], 0
            current.append(sent.text)
            token_count += stoks
    if current:
        chunks.append({
            "text": " ".join(current),
            "meta": {"last_block_idx": len(elements)-1}
        })
    return chunks
