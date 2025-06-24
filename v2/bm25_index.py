from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk import download as nltk_download
from typing import List

# ------------------------------------------------------------------
# Stop-word list (download at runtime if missing)
# ------------------------------------------------------------------
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk_download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


class BM25Index:
    """Lightweight BM25 with stop-word removal."""

    def __init__(self, docs: List[str]):
        self.docs = docs
        tokenized = [self._tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str):
        return [w for w in text.split() if w.lower() not in STOP_WORDS]

    def query(self, q: str, top_k: int = 5):
        scores = self.bm25.get_scores(self._tokenize(q))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_idx]
