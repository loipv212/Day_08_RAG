"""
Task 5 — Semantic Search Module (dense retrieval).

Dùng lại collection ChromaDB đã index ở Task 4 và cùng embedding model
(paraphrase-multilingual-MiniLM-L12-v2 qua fastembed) để đảm bảo nhất quán.

Cơ chế:
    1. Embed query bằng đúng model ở Task 4.
    2. Query ChromaDB theo cosine distance.
    3. Đổi distance → similarity score = 1 - distance, sort giảm dần.
"""

from src.task4_chunking_indexing import embed_texts, query_store, store_count


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa (dense) trên vector store.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending (cosine similarity).
    """
    if store_count() == 0:
        return []
    query_emb = embed_texts([query])[0]
    return query_store(query_emb, top_k=top_k)


if __name__ == "__main__":
    for r in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata'].get('source')}) {r['content'][:90]}...")
