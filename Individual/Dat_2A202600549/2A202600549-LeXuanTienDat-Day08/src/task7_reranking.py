"""
Task 7 — Reranking Module.

Triển khai 3 phương pháp (đều giải thích được cơ chế trong demo):

  1. cross_encoder — chấm lại độ liên quan query↔doc.
       * Nếu có JINA_API_KEY: gọi Jina Reranker v2 (cross-encoder multilingual thật sự).
       * Nếu KHÔNG có key: fallback OFFLINE — kết hợp điểm retrieval gốc (đã chuẩn hoá)
         với "độ phủ từ khoá" (bao nhiêu % token của query xuất hiện trong doc).
         → không cần mạng/API, deterministic, vẫn đẩy doc liên quan lên trên.

  2. mmr — Maximal Marginal Relevance: cân bằng relevance và đa dạng (giảm trùng lặp).
       MMR = λ·sim(q,d) − (1−λ)·max sim(d, đã_chọn).

  3. rrf — Reciprocal Rank Fusion: gộp nhiều ranked list (vd: semantic + lexical).
       RRF(d) = Σ 1/(k + rank_r(d)),  k=60 (Cormack et al. 2009).

Default method = "cross_encoder" (tự chọn offline nếu thiếu key) → Task 9 dùng cái này.
"""

import os
import re

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _minmax_norm(values: list[float]) -> list[float]:
    """Chuẩn hoá min-max về [0,1] (nếu mọi giá trị bằng nhau → 0.5)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


# ============================================================================
# 1. Cross-encoder (Jina API) + fallback offline
# ============================================================================
def _rerank_offline(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank offline: 0.5*điểm_gốc_chuẩn_hoá + 0.5*độ_phủ_từ_khoá."""
    q_tokens = _tokenize(query)
    base = _minmax_norm([c.get("score", 0.0) for c in candidates])

    rescored = []
    for c, b in zip(candidates, base):
        d_tokens = _tokenize(c["content"])
        coverage = len(q_tokens & d_tokens) / len(q_tokens) if q_tokens else 0.0
        new_score = 0.5 * b + 0.5 * coverage
        rescored.append({**c, "score": new_score})

    rescored.sort(key=lambda r: r["score"], reverse=True)
    return rescored[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank bằng cross-encoder (Jina API) — fallback offline nếu không có key."""
    if not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        return _rerank_offline(query, candidates, top_k)

    try:
        resp = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return [
            {**candidates[r["index"]], "score": float(r["relevance_score"])}
            for r in results
        ]
    except Exception:
        # Lỗi mạng/API → vẫn an toàn nhờ fallback offline
        return _rerank_offline(query, candidates, top_k)


# ============================================================================
# 2. MMR
# ============================================================================
def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance — candidates cần có key 'embedding'."""
    if not candidates:
        return []
    remaining = list(range(len(candidates)))
    selected: list[int] = []

    while remaining and len(selected) < top_k:
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx].get("embedding")
            relevance = _cosine(query_embedding, emb) if emb is not None else \
                candidates[idx].get("score", 0.0)
            max_sim = 0.0
            for sel in selected:
                e1, e2 = candidates[idx].get("embedding"), candidates[sel].get("embedding")
                if e1 is not None and e2 is not None:
                    max_sim = max(max_sim, _cosine(e1, e2))
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[i], "score": candidates[i].get("score", 0.0)} for i in selected]


# ============================================================================
# 3. RRF
# ============================================================================
def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion — gộp nhiều ranked list."""
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for content, score in ranked[:top_k]:
        results.append({**content_map[content], "score": score})
    return results


# ============================================================================
# Unified interface
# ============================================================================
def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Giao diện rerank thống nhất (mặc định cross_encoder, offline-safe)."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # cần query_embedding trong candidates; ở đây fallback offline cho tiện
        return _rerank_offline(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hướng dẫn lập trình Python", "score": 0.6, "metadata": {}},
    ]
    for r in rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2):
        print(f"[{r['score']:.3f}] {r['content']}")
