"""
Task 4 — Chunking & Indexing vào Vector Store.

LỰA CHỌN & LÝ DO (tóm tắt cho buổi demo):

1. Chunking — RecursiveCharacterTextSplitter (langchain-text-splitters)
   - chunk_size = 800 ký tự: văn bản luật chia theo Điều/Khoản; 800 ký tự đủ chứa
     trọn một khoản ngắn hoặc phần đầu một điều, không quá dài gây loãng ngữ nghĩa.
   - chunk_overlap = 120: giữ ngữ cảnh nối giữa 2 chunk (tránh cắt mất câu/điều khoản).
   - separators ưu tiên xuống dòng kép → dòng → câu → từ, nên cắt ở ranh giới tự nhiên.

2. Embedding — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (qua fastembed/ONNX)
   - 384 chiều, ĐA NGÔN NGỮ (hỗ trợ tiếng Việt) — phù hợp dữ liệu luật + báo tiếng Việt.
   - Chạy LOCAL bằng ONNX (không cần torch, không cần API key, miễn phí).
   - Chỉ ~0.22 GB nên tải nhanh, hợp máy yếu.

3. Vector store — ChromaDB (local, persistent)
   - Đơn giản, không cần Docker/Cloud; lưu xuống đĩa (chroma_db/) để Task 5/6/9 dùng lại.
   - Lưu sẵn cả document gốc → Task 6 (BM25) đọc lại corpus từ chính collection này.

Cài đặt:
    pip install langchain-text-splitters fastembed chromadb
"""

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
STORE_DIR = PROJECT_DIR / "vector_store"
STORE_EMB = STORE_DIR / "embeddings.npy"
STORE_ITEMS = STORE_DIR / "items.json"

# =============================================================================
# CONFIGURATION
# =============================================================================
CHUNK_SIZE = 800            # ký tự — đủ chứa 1 khoản/đoạn luật, không quá dài
CHUNK_OVERLAP = 120         # giữ ngữ cảnh nối giữa các chunk
CHUNKING_METHOD = "recursive"

# Embedding provider: "openai" (cần OPENAI_API_KEY) hoặc "local" (fastembed, miễn phí).
# Chọn OpenAI text-embedding-3-small: 1536 chiều, đa ngôn ngữ tốt (cả tiếng Việt),
# không phải tải model nặng; chạy nhanh, chi phí rất thấp.
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# (Phương án local nếu không có key — đổi PROVIDER="local" để dùng 2 dòng dưới)
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_STORE = "numpy-cosine"  # tự code, tránh ChromaDB segfault trên Windows

# =============================================================================
# Embedding (Task 5/6/9 import lại embed_texts để dùng chung)
# =============================================================================
_openai_client = None
_local_embedder = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text → list vector (list[float]). Theo EMBEDDING_PROVIDER."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "openai":
        client = _get_openai_client()
        vectors = []
        B = 100  # batch để tránh quá kích thước request
        for i in range(0, len(texts), B):
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL, input=texts[i:i + B]
            )
            vectors.extend([d.embedding for d in resp.data])
        return vectors
    # local fastembed
    global _local_embedder
    if _local_embedder is None:
        from fastembed import TextEmbedding
        _local_embedder = TextEmbedding(model_name=LOCAL_EMBEDDING_MODEL)
    return [v.tolist() for v in _local_embedder.embed(texts)]


# =============================================================================
# Vector store tự code bằng numpy (cosine similarity).
# Lý do: ChromaDB 1.5.9 gây access-violation (segfault) trên Windows máy này.
# Với ~vài nghìn chunk, numpy dot-product chạy tức thì, không phụ thuộc native lib.
# =============================================================================
_store_cache = None


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def get_store() -> dict:
    """Load store từ đĩa (cache trong RAM). Trả về {'emb': ndarray, 'items': list}."""
    global _store_cache
    if _store_cache is None:
        if STORE_EMB.exists() and STORE_ITEMS.exists():
            emb = np.load(STORE_EMB)
            items = json.loads(STORE_ITEMS.read_text(encoding="utf-8"))
        else:
            emb = np.zeros((0, EMBEDDING_DIM), dtype="float32")
            items = []
        _store_cache = {"emb": emb, "items": items}
    return _store_cache


def store_count() -> int:
    return len(get_store()["items"])


def get_all_chunks() -> list[dict]:
    """Trả về toàn bộ chunk {'content', 'metadata'} (dùng cho BM25 Task 6)."""
    return get_store()["items"]


def query_store(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Cosine search trên store. Trả về list {'content','score','metadata'} sort desc."""
    store = get_store()
    emb = store["emb"]
    if emb.shape[0] == 0:
        return []
    q = np.asarray(query_embedding, dtype="float32")
    q = q / (np.linalg.norm(q) or 1.0)
    sims = emb @ q  # emb đã normalize → tích vô hướng = cosine
    top_idx = np.argsort(-sims)[:top_k]
    return [
        {
            "content": store["items"][i]["content"],
            "score": float(sims[i]),
            "metadata": store["items"][i]["metadata"],
        }
        for i in top_idx
    ]


# =============================================================================
# IMPLEMENTATION
# =============================================================================
def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents bằng RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks, thêm key 'embedding'."""
    texts = [c["content"] for c in chunks]
    for chunk, emb in zip(chunks, embed_texts(texts)):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks (kèm embedding) xuống numpy store. Reindex sạch mỗi lần chạy."""
    global _store_cache
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    emb = _normalize(np.asarray([c["embedding"] for c in chunks], dtype="float32"))
    items = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]

    np.save(STORE_EMB, emb)
    STORE_ITEMS.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    _store_cache = {"emb": emb, "items": items}  # cập nhật cache
    return len(items)


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} @ {STORE_DIR}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding'])})")

    count = index_to_vectorstore(chunks)
    print(f"✓ Indexed {count} chunks vào numpy store @ {STORE_DIR}")


if __name__ == "__main__":
    run_pipeline()
