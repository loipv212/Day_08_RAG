"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        return

    # Ở phiên bản SDK mới (>=0.2.x), class đổi tên thành PageIndexClient
    from pageindex import PageIndexClient
    client = PageIndexClient(PAGEINDEX_API_KEY)

    print(f"Đang upload tài liệu từ thư mục {LANDING_DIR}...")
    
    # Quét qua tất cả file .pdf và upload bằng submit_document
    for pdf_file in LANDING_DIR.rglob("*.pdf"):
        try:
            # SDK mới nhận trực tiếp đường dẫn file
            res = client.submit_document(str(pdf_file))
            doc_id = res.get("id", "unknown_id")
            print(f"  ✓ Upload thành công: {pdf_file.name} (ID: {doc_id})")
        except Exception as e:
            print(f"  ✗ Lỗi khi upload {pdf_file.name}: {e}")
            
    print("✓ Đã upload xong tất cả tài liệu!")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env để thực hiện tìm kiếm.")
        return []

    from pageindex import PageIndexClient
    client = PageIndexClient(PAGEINDEX_API_KEY)
    
    try:
        # SDK mới sử dụng chat_completions theo chuẩn OpenAI
        # Truyền doc_id = None để search trên toàn bộ tài liệu đã upload
        response = client.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=None,
            enable_citations=True
        )
        
        # Parse kết quả trả về từ API
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        return [
            {
                "content": content,
                "score": 1.0,  # PageIndex API không trả về score tương tự vector
                "metadata": {"source": "pageindex_vectorless"},
                "source": "pageindex"
            }
        ]
    except Exception as e:
        print(f"⚠ Lỗi khi tìm kiếm qua PageIndex API: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ CHÚ Ý: Tính năng Vectorless RAG này yêu cầu có API Key của PageIndex.")
        print("  1. Hãy truy cập trang: https://pageindex.ai/ để đăng ký.")
        print("  2. Lấy API Key và dán vào file .env với cú pháp: PAGEINDEX_API_KEY=your_key_here")
    else:
        # Nếu muốn upload lại file, hãy bỏ comment dòng dưới đây
        upload_documents()

        print("\nTest query:")
        test_query = "hình phạt sử dụng ma tuý"
        print(f"Query: '{test_query}'")
        
        results = pageindex_search(test_query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f} | Source: {r['metadata'].get('source')}")
            print(f"    Content: {r['content'][:200]}...\n")
