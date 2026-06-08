"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store: thay vì chunk + embed +
so khớp vector, PageIndex phân tích cấu trúc tài liệu (mục lục/section) thành
một cây (tree), rồi dùng LLM để duyệt cây và chọn ra (các) node liên quan nhất
tới câu hỏi — gọi là "vectorless" vì không có bước embedding/ANN search nào.

Cài đặt:
    pip install pageindex

Cách dùng (cần PAGEINDEX_API_KEY trong file .env, lấy tại pageindex.ai):
    1. upload_documents() — submit các PDF luật trong data/landing/legal/,
       PageIndex xử lý OCR + sinh cây cấu trúc (tree generation, có thể mất
       vài phút với văn bản dài). doc_id được lưu vào data/pageindex_docs.json
       để các lần chạy sau tái sử dụng, không upload lại.
    2. pageindex_search(query) — submit_query trên từng doc_id đã upload,
       poll get_retrieval tới khi status="completed", gom retrieved_nodes
       thành list kết quả thống nhất với schema của Task 5/6 (có thêm
       'source': 'pageindex' để đánh dấu nguồn — dùng khi merge ở Task 9).

LƯU Ý: PageIndex chỉ nhận PDF (submit_document upload file nhị phân để OCR +
sinh cây), nên dùng trực tiếp PDF gốc trong data/landing/legal/ thay vì bản
.md đã chuẩn hoá ở Task 3 — PageIndex tự lo việc đọc cấu trúc tài liệu.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

# Tree generation cho văn bản luật (vài chục trang) có thể mất nhiều phút vì
# PageIndex chạy OCR + LLM phân tích cấu trúc; retrieval thường nhanh hơn nhiều.
POLL_INTERVAL_SECONDS = 5
UPLOAD_TIMEOUT_SECONDS = 1800
RETRIEVAL_TIMEOUT_SECONDS = 180

_client = None


def _get_client():
    """Khởi tạo PageIndexClient (lazy) — báo lỗi rõ ràng nếu thiếu API key."""
    global _client
    if _client is None:
        if not PAGEINDEX_API_KEY:
            raise RuntimeError(
                "Thiếu PAGEINDEX_API_KEY. Đăng ký tài khoản tại "
                "https://pageindex.ai/, lấy API key rồi thêm dòng "
                "PAGEINDEX_API_KEY=... vào file .env"
            )
        from pageindex import PageIndexClient
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def _load_doc_registry() -> dict[str, str]:
    """Đọc map {filename: doc_id} đã upload từ lần chạy trước (nếu có)."""
    if DOC_REGISTRY_PATH.exists():
        return json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_doc_registry(registry: dict[str, str]) -> None:
    DOC_REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upload_documents() -> dict[str, str]:
    """
    Upload các văn bản luật PDF trong data/landing/legal/ lên PageIndex và
    chờ tới khi tree generation hoàn tất (status == "completed", sẵn sàng
    cho retrieval).

    Returns:
        Dict {filename: doc_id} — cũng được lưu vào data/pageindex_docs.json
        để gọi lại pageindex_search() ở các lần sau không cần upload lại.
    """
    client = _get_client()
    registry = _load_doc_registry()

    for pdf_file in sorted(LEGAL_DIR.glob("*.pdf")):
        if pdf_file.name in registry:
            print(f"  - Bỏ qua (đã có doc_id): {pdf_file.name}")
            continue
        result = client.submit_document(str(pdf_file))
        doc_id = result["doc_id"]
        registry[pdf_file.name] = doc_id
        _save_doc_registry(registry)
        print(f"  + Submitted: {pdf_file.name} -> doc_id={doc_id}")

    pending = set(registry.values())
    start = time.time()
    while pending and time.time() - start < UPLOAD_TIMEOUT_SECONDS:
        for doc_id in list(pending):
            if client.get_tree(doc_id).get("status") == "completed":
                pending.discard(doc_id)
                print(f"  ✓ Sẵn sàng retrieval: {doc_id}")
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)

    if pending:
        print(
            f"  ⚠ {len(pending)} document(s) chưa xử lý xong sau "
            f"{UPLOAD_TIMEOUT_SECONDS}s: {sorted(pending)}"
        )

    return registry


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    """Poll get_retrieval(retrieval_id) tới khi status == 'completed'."""
    start = time.time()
    while time.time() - start < RETRIEVAL_TIMEOUT_SECONDS:
        result = client.get_retrieval(retrieval_id)
        if result.get("status") == "completed":
            return result
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Retrieval {retrieval_id} chưa hoàn tất sau {RETRIEVAL_TIMEOUT_SECONDS}s"
    )


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex — duyệt cây cấu trúc tài liệu bằng
    LLM thay vì so khớp vector embedding. Dùng làm fallback khi hybrid search
    (semantic + lexical, Task 5/6) không trả kết quả đủ tốt (Task 9).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,           # nội dung đoạn liên quan trong node
            'score': float,           # suy ra từ thứ hạng PageIndex trả về
                                      # (PageIndex không trả relevance score
                                      # tường minh — retrieved_nodes đã được
                                      # sắp theo độ liên quan giảm dần, nên
                                      # dùng 1/rank tương tự ý tưởng RRF)
            'metadata': dict,         # {'source', 'doc_id', 'node_id', 'title'}
            'source': 'pageindex'     # đánh dấu nguồn — dùng khi merge Task 9
        }
    """
    client = _get_client()
    registry = _load_doc_registry() or upload_documents()

    results = []
    for filename, doc_id in registry.items():
        submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
        retrieval = _wait_for_retrieval(client, submitted["retrieval_id"])

        for rank, node in enumerate(retrieval.get("retrieved_nodes", []), start=1):
            # API trả "relevant_contents" là list-of-list-of-dict (mỗi nhóm
            # ứng với 1 đoạn matched trong node) — cần duyệt 2 cấp để lấy
            # field "relevant_content" ở dict trong cùng.
            snippets = [
                rc.get("relevant_content", "")
                for group in node.get("relevant_contents", [])
                for rc in group
            ]
            content = "\n".join(s for s in snippets if s) or node.get("title", "")
            results.append({
                "content": content,
                "score": 1.0 / rank,
                "metadata": {
                    "source": filename,
                    "doc_id": doc_id,
                    "node_id": node.get("id"),
                    "title": node.get("title"),
                },
                "source": "pageindex",
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        for r in pageindex_search("hình phạt sử dụng ma tuý", top_k=3):
            print(f"[{r['score']:.3f}] {r['metadata']['title']}: {r['content'][:100]}...")
