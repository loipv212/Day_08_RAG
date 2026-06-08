"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =============================================================================
# Cấu hình phải khớp hoàn toàn với Task 4
# =============================================================================
VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "vector_store" / "chroma"
COLLECTION_NAME = "drug_law_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Load model ở global scope để không bị load lại mỗi lần tìm kiếm
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
model.max_seq_length = 512

# Kết nối ChromaDB
try:
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✓ Connected to ChromaDB. Collection '{COLLECTION_NAME}' has {collection.count()} chunks.\n")
except Exception as e:
    print(f"Lỗi kết nối ChromaDB: {e}")
    collection = None

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (0.0 đến 1.0)
            'metadata': dict     # source, doc_type, chunk_index
        }
    """
    if collection is None:
        print("⚠ Không tìm thấy collection. Hãy đảm bảo Task 4 đã chạy thành công.")
        return []

    # Bước 1: Embed query bằng cùng model ở Task 4 (bật normalize)
    query_embedding = model.encode(
        [query], 
        normalize_embeddings=True
    )[0].tolist()
    
    # Bước 2: Query vector store
    # Mặc định ChromaDB trả về bình phương khoảng cách L2 (L2 squared distance). 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Bước 3: Định dạng kết quả và chuyển đổi khoảng cách sang score
    formatted_results = []
    if not results["documents"] or not results["documents"][0]:
        return formatted_results

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    
    for doc, dist, meta in zip(documents, distances, metadatas):
        # Vì vector đã được normalize (độ dài = 1), công thức quy đổi từ khoảng cách L2 sang Cosine:
        # Cosine Similarity = 1 - (L2_squared_distance / 2)
        score = 1.0 - (dist / 2.0)
        
        formatted_results.append({
            "content": doc,
            "score": float(score),
            "metadata": meta
        })
        
    return formatted_results


if __name__ == "__main__":
    # Test thử 
    test_query = "hình phạt cho tội tàng trữ ma tuý"
    print(f"Đang tìm kiếm cho câu hỏi: '{test_query}'\n")
    print("-" * 60)
    
    results = semantic_search(test_query, top_k=5)
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] Score: {r['score']:.4f} | Source: {r['metadata'].get('source')}")
        # In 200 ký tự đầu tiên để xem qua nội dung
        snippet = r['content'][:200].replace('\n', ' ')
        print(f"    Content: {snippet}...\n")
