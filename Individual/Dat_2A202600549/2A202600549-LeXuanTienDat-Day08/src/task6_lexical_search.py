"""
Task 6 — Lexical Search Module (BM25).

Dùng BM25Okapi (rank-bm25). Corpus được đọc lại từ chính collection ChromaDB
đã index ở Task 4 → đảm bảo lexical và semantic search chạy trên CÙNG tập chunk.

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao.
    - Inverse Document Frequency (IDF): từ hiếm (đặc trưng) → trọng số lớn hơn.
    - Chuẩn hoá độ dài: document dài không bị ưu tiên quá mức.
    - score(q,d) = Σ IDF(qi) * (tf*(k1+1)) / (tf + k1*(1-b+b*|d|/avgdl)), k1=1.5, b=0.75.

Tokenize: tiếng Việt tách theo khoảng trắng + bỏ dấu câu (đơn giản, không cần
underthesea). Đủ tốt cho BM25 vì BM25 dựa trên trùng khớp token bề mặt.
"""

import re

from rank_bm25 import BM25Okapi

from src.task4_chunking_indexing import get_all_chunks

# Cache: corpus + index BM25 (build 1 lần)
_corpus: list[dict] | None = None
_bm25: BM25Okapi | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase, tách từ, bỏ ký tự không phải chữ/số."""
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return tokens


def _load_corpus() -> list[dict]:
    """Đọc toàn bộ chunk từ numpy store của Task 4."""
    return [
        {"content": c["content"], "metadata": c.get("metadata", {})}
        for c in get_all_chunks()
    ]


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Xây dựng BM25 index từ corpus."""
    tokenized = [_tokenize(d["content"]) for d in corpus]
    return BM25Okapi(tokenized)


def _ensure_index():
    global _corpus, _bm25
    if _bm25 is None:
        _corpus = _load_corpus()
        if _corpus:
            _bm25 = build_bm25_index(_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khoá bằng BM25.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending (chỉ giữ score > 0).
    """
    _ensure_index()
    if not _corpus or _bm25 is None:
        return []

    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": _corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": _corpus[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    for r in lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata'].get('source')}) {r['content'][:90]}...")
