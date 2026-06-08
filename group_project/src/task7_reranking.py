"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lựa chọn cho pipeline này: MMR (mặc định, dùng làm rerank chính qua giao
diện `rerank()`) — vì `sentence_transformers.CrossEncoder` bị Segmentation
fault trên môi trường CPU/torch của máy này (đã thử cả
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 lẫn cross-encoder/ms-marco-MiniLM-L-6-v2,
cả hai đều crash khi load model — lỗi môi trường, không phải lỗi code).
MMR dùng cùng OpenAI embedding model với Task 4/5 (`embed_texts`, đảm bảo
query và candidate nằm cùng không gian vector — không cần load thêm model
local nào), và còn có lợi thế giảm trùng lặp nội dung giữa các kết quả top-k
(quan trọng với corpus có nhiều đoạn lặp ý — vd. nhiều bài báo cùng nói về
1 vụ án). `rerank_cross_encoder` vẫn được implement đầy đủ để tham khảo,
nhưng không dùng làm default vì lý do môi trường nêu trên.
"""

from typing import Optional

# Cross-encoder reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 — model
# multilingual MiniLM huấn luyện trên mMARCO (gồm cả tiếng Việt), chạy local
# (không cần API key như Jina), nhẹ (~470MB) nên ổn trên CPU. Khác với
# bi-encoder ở Task 5 (encode query/doc riêng rồi so cosine), cross-encoder
# encode đồng thời cặp (query, document) qua cùng 1 model nên nắm bắt tương
# tác token-level tốt hơn → relevance score chính xác hơn, đánh đổi bằng
# việc chạy chậm hơn (không thể pre-compute embedding cho document).
# LƯU Ý: model này gây Segmentation fault khi load trên môi trường hiện tại
# (lỗi torch/CrossEncoder cấp native, đã verify với 2 model khác nhau) nên
# không dùng làm phương pháp mặc định — xem giải thích ở module docstring.
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    reranked = [
        {**c, "score": float(score)}
        for c, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    import numpy as np

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[i], "score": float(cosine_sim(query_embedding, candidates[i]["embedding"]))}
            for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Embed query + candidates bằng cùng OpenAI embedding model với Task 4/5
        # (đảm bảo cùng không gian vector), rồi áp dụng MMR để vừa relevant
        # vừa đa dạng (giảm trùng lặp nội dung).
        from .task4_chunking_indexing import embed_texts
        texts = [query] + [c["content"] for c in candidates]
        embeddings = embed_texts(texts)
        query_embedding, candidate_embeddings = embeddings[0], embeddings[1:]
        enriched = [{**c, "embedding": emb} for c, emb in zip(candidates, candidate_embeddings)]
        return rerank_mmr(query_embedding, enriched, top_k=top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
