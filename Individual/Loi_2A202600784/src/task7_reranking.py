"""
Task 7 — Reranking Module.

Cung cấp các phương pháp Reranking (Xếp hạng lại) cho danh sách kết quả tìm kiếm:
1. Cross-encoder: Chấm điểm lại độ liên quan cực kỳ chính xác.
2. MMR: Đa dạng hóa kết quả.
3. RRF: Gộp kết quả từ nhiều công cụ tìm kiếm khác nhau.
"""

from typing import Optional
import numpy as np

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Local).
    """
    print("Đang chạy Rerank bằng Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
    from sentence_transformers import CrossEncoder
    
    # Khởi tạo model (trong thực tế nên đưa ra ngoài global để khỏi load lại)
    # BAAI/bge-reranker-v2-m3 hỗ trợ đa ngôn ngữ cực tốt
    model = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    pairs = [[query, doc["content"]] for doc in candidates]
    scores = model.predict(pairs)
    
    reranked = []
    for doc, score in zip(candidates, scores):
        doc_copy = doc.copy()
        doc_copy["score"] = float(score)
        reranked.append(doc_copy)
        
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def cosine_sim(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    print(f"Đang chạy Rerank bằng MMR (lambda={lambda_param})...")
    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score formula
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (Ví dụ: Semantic Search + BM25).
    """
    print(f"Đang chạy Rerank bằng RRF (k={k})...")
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            # Công thức RRF: Cộng dồn 1 / (60 + Thứ hạng)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    # Sắp xếp lại dựa trên điểm RRF dồn
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder", 
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Hãy gọi thẳng hàm rerank_mmr() vì nó cần truyền vào mảng query_embedding")
    elif method == "rrf":
        raise NotImplementedError("Hãy gọi thẳng hàm rerank_rrf() vì nó cần truyền vào danh sách mảng kết quả ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST RERANKING MODULE")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # TEST 1: RRF
    # ---------------------------------------------------------
    # Giả lập kết quả từ Semantic Search (Task 5)
    list_from_semantic = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "metadata": {"source": "luat1.md"}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "metadata": {"source": "luat2.md"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "metadata": {"source": "news1.md"}}
    ]
    
    # Giả lập kết quả từ BM25 (Task 6)
    list_from_bm25 = [
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "metadata": {"source": "news1.md"}},
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "metadata": {"source": "luat1.md"}},
        {"content": "Bắt quả tang tàng trữ 5kg Fentanyl", "metadata": {"source": "news2.md"}}
    ]
    
    print("\n[ TEST 1: Dùng RRF để gộp 2 danh sách lại với nhau ]")
    rrf_results = rerank_rrf([list_from_semantic, list_from_bm25], top_k=3)
    for i, r in enumerate(rrf_results, 1):
        print(f"[{i}] RRF Score: {r['score']:.4f} | {r['content']}")

    # ---------------------------------------------------------
    # TEST 2: Cross-Encoder
    # ---------------------------------------------------------
    print("\n[ TEST 2: Dùng Cross-Encoder chấm điểm lại ]")
    # Lấy 3 kết quả từ Test 1 đem đi chấm lại bằng AI
    ce_results = rerank("tàng trữ ma tuý phạt như nào", rrf_results, top_k=2, method="cross_encoder")
    for i, r in enumerate(ce_results, 1):
        print(f"[{i}] CE Score: {r['score']:.4f} | {r['content']}")
