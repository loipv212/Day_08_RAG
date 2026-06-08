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

import sys
from pathlib import Path

# Thêm thư mục hiện tại vào sys.path để có thể import các module dễ dàng khi chạy trực tiếp
sys.path.append(str(Path(__file__).parent))

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.5   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.
    """
    print(f"\n[PIPELINE] Bắt đầu tìm kiếm: '{query}'")
    
    # Step 1: Song song chạy semantic + lexical
    print("  ├─ Bước 1: Gọi Semantic Search (Vector) và Lexical Search (BM25)...")
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)
    
    # Step 2: Merge bằng RRF
    print("  ├─ Bước 2: Trộn 2 danh sách kết quả bằng RRF...")
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid_search"
        
    # Step 3: Rerank
    if use_reranking and merged:
        print(f"  ├─ Bước 3: Đưa AI ({RERANK_METHOD}) vào đọc và chấm điểm lại toàn bộ...")
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        print("  ├─ Bước 3: Bỏ qua AI Reranking.")
        final_results = merged[:top_k]
        
    # Step 4: Check threshold → fallback
    best_score = final_results[0]["score"] if final_results else 0
    if not final_results or best_score < score_threshold:
        print(f"  ⚠ Điểm độ tin cậy quá thấp ({best_score:.3f} < {score_threshold}). Chắc RAG tìm không ra.")
        print("  └─ Bước 4: Chuyển sang phương án B (gọi thử API PageIndex)...")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback
        else:
            print("  ⚠ Fallback PageIndex thất bại (có thể do API/Credit). Trả về kết quả Hybrid hiện tại.")
    else:
        print(f"  └─ Hoàn thành! Tìm được câu trả lời tốt (Score: {best_score:.3f})")
        
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
    ]

    for q in test_queries:
        print("=" * 80)
        results = retrieve(q, top_k=3)
        print("-" * 80)
        for i, r in enumerate(results, 1):
            source = r.get('source', 'unknown')
            file_name = r.get('metadata', {}).get('source', 'unknown')
            score = r.get('score', 0)
            
            # Format text in ra cho gọn gàng
            content = r['content'].replace('\n', ' ')
            print(f"  {i}. [{score:.3f}] [Nguồn: {source}] [File: {file_name}]\n     {content[:150]}...\n")
