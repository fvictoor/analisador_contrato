from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def _chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    # Divide o texto em blocos de ~max_chars, respeitando quebras simples
    chunks: List[str] = []
    buf = []
    current_len = 0
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            continue
        if current_len + len(p) + 1 > max_chars:
            if buf:
                chunks.append("\n".join(buf))
                buf = []
                current_len = 0
        buf.append(p)
        current_len += len(p) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def retrieve_relevant_chunks(question: str, text: str, top_k: int = 5) -> List[str]:
    chunks = _chunk_text(text, max_chars=1400)
    if not chunks:
        return []
    corpus = chunks + [question]
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(corpus)
    question_vec = tfidf[-1]
    doc_vectors = tfidf[:-1]
    sims = linear_kernel(question_vec, doc_vectors).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_idx]