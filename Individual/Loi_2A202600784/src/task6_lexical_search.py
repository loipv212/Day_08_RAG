"""
Task 6 — Lexical Search Module (BM25).

Cơ chế hoạt động của BM25:
- Tách từng từ trong văn bản ra (Tokenization).
- Term Frequency (TF): Đếm số lần từ khóa xuất hiện trong văn bản. Càng nhiều -> càng liên quan (điểm cao).
- Inverse Document Frequency (IDF): Nếu từ khóa đó quá phổ biến ở mọi văn bản (ví dụ chữ "và", "là", "của") thì độ quan trọng của từ đó sẽ bị giảm xuống. Từ nào hiếm gặp (ví dụ "248", "Fentanyl") thì sẽ có trọng số cao hơn.
- Document Length Normalization: Chống thiên vị các văn bản dài. Nếu 2 văn bản có số lần nhắc đến từ khóa bằng nhau, văn bản ngắn hơn sẽ được điểm cao hơn vì nó đi trực tiếp vào vấn đề.
"""

from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
import re

VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "vector_store" / "chroma"
COLLECTION_NAME = "drug_law_docs"

# 1. Load corpus trực tiếp từ ChromaDB (để đảm bảo đồng nhất với Task 4 & 5)
print("Loading corpus from ChromaDB...")
try:
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas"])

    CORPUS = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        CORPUS.append({
            "content": doc,
            "metadata": meta
        })
    print(f"✓ Loaded {len(CORPUS)} chunks from database.")
except Exception as e:
    print(f"⚠ Không thể load dữ liệu từ ChromaDB: {e}")
    CORPUS = []

def tokenize(text: str) -> list[str]:
    """Hàm tách từ cơ bản bằng Regular Expression (loại bỏ dấu câu)"""
    # Lấy các từ chứa chữ cái và số, bỏ qua các dấu chấm, phẩy, gạch ngang...
    return re.findall(r'\w+', text.lower())

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.
    """
    if not corpus:
        return None
    print("Building BM25 index... (this might take a moment)")
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("✓ BM25 index built successfully.\n")
    return bm25_index

# Xây dựng Index ngay khi load file
bm25 = build_bm25_index(CORPUS)

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    """
    if bm25 is None:
        return []

    # Bước 1: Tách từ câu hỏi bằng cùng 1 hàm tokenize
    tokenized_query = tokenize(query)
    
    # Bước 2: Dùng BM25 để lấy mảng điểm số cho toàn bộ corpus
    scores = bm25.get_scores(tokenized_query)
    
    # Bước 3: Lấy ra top_k index có điểm số cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        # Chỉ lấy những kết quả có điểm > 0 (nghĩa là có ít nhất 1 từ khóa trùng khớp)
        if score > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": score,
                "metadata": CORPUS[idx]["metadata"]
            })
            
    return results


if __name__ == "__main__":
    test_query = "Điều 248 tàng trữ trái phép chất ma tuý"
    print("-" * 60)
    print(f"Test Task 6 (BM25 Lexical Search)\nQuery: '{test_query}'\n")
    
    results = lexical_search(test_query, top_k=5)
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] BM25 Score: {r['score']:.4f} | Source: {r['metadata'].get('source')}")
        snippet = r['content'][:200].replace('\n', ' ')
        print(f"    Content: {snippet}...\n")
