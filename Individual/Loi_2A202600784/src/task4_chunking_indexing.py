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
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import shutil

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "vector_store" / "chroma"
COLLECTION_NAME = "drug_law_docs"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# CHUNKING STRATEGY: Chọn RecursiveCharacterTextSplitter thay vì MarkdownHeaderTextSplitter
# Vì các file luật (73luat.md) không sử dụng thẻ Header (#) của Markdown cho các Điều luật, 
# và các file báo (article_xx.md) chứa nhiều Header rác do quá trình crawl.
CHUNK_SIZE = 500        # Vì sao chọn 500? Độ dài vừa đủ để chứa trọn vẹn ngữ nghĩa của một Điều luật hoặc một đoạn báo mà không bị loãng.
CHUNK_OVERLAP = 50      # Vì sao chọn 50? Giúp các câu dài không bị đứt đoạn gãy ý giữa 2 chunk.
CHUNKING_METHOD = "recursive"

# EMBEDDING MODEL
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Đây là mô hình đa ngôn ngữ hỗ trợ tiếng Việt rất tốt, vượt trội hơn all-MiniLM-L6-v2 (vốn thiên về tiếng Anh).
EMBEDDING_DIM = 1024

# VECTOR STORE
# Chọn ChromaDB thay cho Weaviate vì ChromaDB dễ dàng cài đặt và lưu thẳng ra file ở máy (local)
# mà không cần thiết lập Docker hay tài khoản Cloud phức tạp, cực kỳ phù hợp cho môi trường dev.
VECTOR_STORE = "chromadb"


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
        try:
            content = md_file.read_text(encoding="utf-8").strip()

            if not content:
                print(f"⚠ Bỏ qua file rỗng: {md_file}")
                continue

            relative_path = md_file.relative_to(STANDARDIZED_DIR)
            doc_type = "legal" if "legal" in relative_path.parts else "news"

            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(relative_path).replace("\\", "/")
                }
            })

        except Exception as e:
            print(f"✗ Lỗi đọc file {md_file}: {e}")

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

    for doc_id, doc in enumerate(documents):
        splits = splitter.split_text(doc["content"])

        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            chunk_id = f"{doc['metadata']['type']}_{doc_id:03d}_chunk_{i:04d}"

            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_id": chunk_id
                }
            })

    return chunks

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    if not chunks:
        print("⚠ Không có chunk nào để embed.")
        return chunks

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 512  # Giới hạn số token tối đa để không bị văng VRAM đột ngột
    texts = [c["content"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=4,  # Hạ batch_size xuống để GPU (RTX 3050 4GB) không bị tràn VRAM (Out of Memory)
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE != "chromadb":
        raise ValueError(f"Vector store chưa được hỗ trợ trong file này: {VECTOR_STORE}")

    import chromadb

    if not chunks:
        print("⚠ Không có chunk nào để index.")
        return

    # Xóa vector store cũ để tránh trùng dữ liệu khi chạy lại nhiều lần
    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "RAG documents about drug law, news, and prevention"
        }
    )

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["content"])
        embeddings.append(chunk["embedding"])

        metadata = chunk["metadata"]

        metadatas.append({
            "source": str(metadata.get("source", "")),
            "type": str(metadata.get("type", "")),
            "path": str(metadata.get("path", "")),
            "doc_id": int(metadata.get("doc_id", 0)),
            "chunk_index": int(metadata.get("chunk_index", 0)),
            "chunk_id": str(metadata.get("chunk_id", "")),
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"✓ Saved ChromaDB at: {VECTOR_DB_DIR}")
    print(f"✓ Collection name: {COLLECTION_NAME}")
    print(f"✓ Total indexed chunks: {collection.count()}")

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
