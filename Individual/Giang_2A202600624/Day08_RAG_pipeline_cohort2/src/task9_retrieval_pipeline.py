"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
# "cross_encoder" gây Segmentation fault trên môi trường này (xem giải thích ở
# module docstring của Task 7) nên dùng "mmr" — vừa ổn định vừa trả về score là
# cosine similarity (0..1), so sánh được trực tiếp với SCORE_THRESHOLD.
RERANK_METHOD = "mmr"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Chạy semantic (dense) + lexical (sparse) search độc lập, lấy dư
    # top_k*2 ứng viên mỗi nhánh để RRF có đủ "nguyên liệu" gộp đa dạng.
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge 2 ranked list bằng Reciprocal Rank Fusion — gộp theo thứ
    # hạng (không phụ thuộc thang điểm khác nhau giữa cosine similarity và
    # BM25), kết quả xuất hiện ở cả 2 nhánh sẽ được ưu tiên cao hơn.
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank để có điểm relevance trên cùng 1 thang đo (cosine
    # similarity, 0..1) — so sánh được trực tiếp với score_threshold ở bước 4.
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item.setdefault("source", "hybrid")
    else:
        final_results = merged[:top_k]

    # Step 4: Nếu hybrid không tìm được gì đủ liên quan (best score thấp hơn
    # ngưỡng, hoặc rỗng) → fallback sang PageIndex (duyệt cây cấu trúc bằng LLM,
    # có thể tìm ra ngữ cảnh mà semantic/lexical bỏ sót do giới hạn chunk).
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback → PageIndex"
        )
        try:
            return pageindex_search(query, top_k=top_k)
        except Exception as e:
            # Fallback chỉ nên "tốt hơn nếu có", không được phép biến cả pipeline
            # thành crash khi PageIndex tạm thời không dùng được (hết credit,
            # mạng, timeout...) — trả về kết quả hybrid hiện có (dù dưới ngưỡng)
            # còn hơn không trả gì.
            print(f"  ⚠ PageIndex fallback lỗi ({e}) — dùng tạm kết quả hybrid")
            return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
