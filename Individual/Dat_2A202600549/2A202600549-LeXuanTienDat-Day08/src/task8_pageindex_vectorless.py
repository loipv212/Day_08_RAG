"""
Task 8 — PageIndex Vectorless RAG.

PageIndex (https://pageindex.ai) làm RAG KHÔNG cần vector store: nó xây "cây cấu trúc"
(tree) của tài liệu rồi reasoning để tìm phần liên quan — thay vì embedding + cosine.

Lưu ý: endpoint /retrieval/ cũ đã DEPRECATED. Bản này dùng **Chat-completion API mới**
(POST /chat/completions, kiểu OpenAI) với `doc_id` + `enable_citations` → trả về câu trả
lời đã được PageIndex truy xuất + tổng hợp, kèm citation dạng <doc=file.pdf;page=N>.

Quy trình:
    1. upload_documents(): upload các PDF luật lên PageIndex, poll tới khi xử lý xong,
       cache {filename: doc_id} vào vector_store/pageindex_docs.json (khỏi upload lại).
    2. pageindex_search(): hỏi chat-completion trên toàn bộ doc_id → trả kết quả
       (đánh dấu source='pageindex'), dùng làm FALLBACK ở Task 9.
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.pageindex.ai"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

PROJECT_DIR = Path(__file__).parent.parent
LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
DOC_CACHE = PROJECT_DIR / "vector_store" / "pageindex_docs.json"


def _headers(json_body: bool = False) -> dict:
    h = {"api_key": PAGEINDEX_API_KEY}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


def _wait_doc_ready(doc_id: str, timeout: int = 300) -> bool:
    """Poll trạng thái xử lý document tới khi 'completed'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{BASE_URL}/doc/{doc_id}/", headers=_headers(), timeout=30)
        status = r.json().get("status")
        if status == "completed":
            return True
        if status == "failed":
            return False
        time.sleep(5)
    return False


def upload_documents(force: bool = False) -> dict:
    """Upload các PDF luật lên PageIndex, trả về {filename: doc_id} và cache xuống đĩa."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    cache = {}
    if DOC_CACHE.exists() and not force:
        cache = json.loads(DOC_CACHE.read_text(encoding="utf-8"))

    pdfs = sorted(LEGAL_DIR.glob("*.pdf"))
    for pdf in pdfs:
        if pdf.name in cache:
            print(f"  ✓ (cache) {pdf.name} -> {cache[pdf.name]}")
            continue
        with open(pdf, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/doc/", headers=_headers(), files={"file": f}, timeout=120
            )
        r.raise_for_status()
        doc_id = r.json()["doc_id"]
        print(f"  ↑ Uploaded {pdf.name} -> {doc_id} (đang xử lý...)")
        _wait_doc_ready(doc_id)
        cache[pdf.name] = doc_id

    DOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DOC_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def _get_doc_ids() -> list[str]:
    """Lấy danh sách doc_id (upload nếu chưa có cache)."""
    if DOC_CACHE.exists():
        cache = json.loads(DOC_CACHE.read_text(encoding="utf-8"))
    else:
        cache = upload_documents()
    return list(cache.values())


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval bằng PageIndex (chat-completion API).
    Dùng làm fallback khi hybrid search không đủ tốt.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env")

    doc_ids = _get_doc_ids()
    if not doc_ids:
        return []

    body = {
        "messages": [{"role": "user", "content": query}],
        "doc_id": doc_ids,
        "enable_citations": True,
    }
    r = requests.post(
        f"{BASE_URL}/chat/completions", headers=_headers(json_body=True),
        json=body, timeout=180,
    )
    r.raise_for_status()
    data = r.json()

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    citations = data.get("citations", [])
    if not answer:
        return []

    return [{
        "content": answer,
        "score": 1.0,  # PageIndex trả 1 câu trả lời tổng hợp (không có cosine score)
        "metadata": {"source": "pageindex", "citations": citations},
        "source": "pageindex",
    }][:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env (đăng ký tại pageindex.ai)")
    else:
        print("Uploading documents lên PageIndex...")
        upload_documents()
        print("\nTest query:")
        for r in pageindex_search("hình phạt tàng trữ trái phép chất ma tuý", top_k=2):
            print(f"[{r['source']}] {r['content'][:200]}...")
            print("citations:", r["metadata"]["citations"][:3])
