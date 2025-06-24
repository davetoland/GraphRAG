# expansion/expander.py
from llama_cpp import Llama
from config import LLAMA_MODEL_PATH, LLAMA_USE_GPU, LLAMA_KWARGS
from ner_kg.ner import extract_entities
from ner_kg.graph_builder import G

# initialize once
llm = Llama(
    model_path=str(LLAMA_MODEL_PATH),
    use_gpu=LLAMA_USE_GPU,
    **LLAMA_KWARGS
)

def expand_query(text: str) -> str:
    ents    = [e for e,_ in extract_entities(text)]
    related = set().union(*(G.neighbors(e) for e in ents))
    prompt  = (
        f"Paraphrase & broaden: {text}\n"
        f"Include terms: {', '.join(related)}"
    )
    resp = llm(prompt=prompt)
    return resp["choices"][0]["text"]
