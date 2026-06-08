"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)  <- đã chọn, cần OPENAI_API_KEY trong .env

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters openai weaviate-client
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking: RecursiveCharacterTextSplitter — tách theo thứ tự ưu tiên các
# separator (đoạn văn -> dòng -> câu -> từ), an toàn và phổ biến cho văn bản
# hỗn hợp (luật + tin tức) mà không cần hiểu cấu trúc heading đặc thù.
# CHUNK_SIZE=500 ký tự: đủ lớn để giữ trọn vẹn ngữ cảnh 1 điều/đoạn ngắn,
# nhưng đủ nhỏ để embedding model (giới hạn ~256-512 token) không bị cắt cụt.
# CHUNK_OVERLAP=50 (10% size): giữ ngữ cảnh nối giữa 2 chunk liền kề, tránh
# việc một câu quan trọng bị cắt đứt ngay ranh giới chunk.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding: OpenAI text-embedding-3-small — dùng API (có OPENAI_API_KEY
# trong .env) thay vì model local. Model multilingual, hỗ trợ tốt tiếng Việt,
# 1536 chiều, chất lượng embedding cao hơn các bi-encoder mở nhỏ (MiniLM) và
# không tốn RAM/CPU để load model lên máy — đánh đổi bằng việc cần gọi API
# (có chi phí theo token, cần mạng) cho mỗi lần embed.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Vector store: ChromaDB — embedded, chạy local (PersistentClient lưu xuống
# disk), không cần dựng server/Docker hay tài khoản cloud như Weaviate, phù
# hợp cho một pipeline chạy trên máy cá nhân.
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


_openai_client = None


def _get_openai_client():
    """Lazy init OpenAI client — dùng chung cho indexing (Task 4) và truy vấn
    (Task 5/7) để đảm bảo query và document nằm cùng không gian embedding."""
    global _openai_client
    if _openai_client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "Thiếu OPENAI_API_KEY trong .env — cần để gọi "
                f"OpenAI embeddings API (model={EMBEDDING_MODEL})"
            )
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed danh sách text bằng OpenAI embeddings API (model=EMBEDDING_MODEL).
    Gửi theo batch để tránh vượt giới hạn request size của API.
    """
    client = _get_openai_client()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset collection để tránh trùng id khi chạy lại pipeline nhiều lần.
    # Dùng cosine space (chuẩn cho semantic search với sentence embeddings)
    # thay vì l2 mặc định, để distance -> similarity score nhất quán.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]
    documents_text = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [
        {"source": c["metadata"]["source"], "type": c["metadata"]["type"],
         "chunk_index": c["metadata"]["chunk_index"]}
        for c in chunks
    ]

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents_text[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
