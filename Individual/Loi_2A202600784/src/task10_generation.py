"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

# Thêm thư mục hiện tại vào sys.path để có thể import các module dễ dàng khi chạy trực tiếp
sys.path.append(str(Path(__file__).parent))

from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là một chuyên gia pháp lý AI chuyên nghiệp. Nhiệm vụ của bạn là tổng hợp các tài liệu được cung cấp và trả lời câu hỏi một cách toàn diện, tự nhiên và mạch lạc nhất bằng tiếng Việt.

QUY TẮC BẮT BUỘC:
1. TRÍCH DẪN NGUỒN: Mọi thông tin, dữ kiện đưa ra ĐỀU PHẢI có trích dẫn nguồn ngay bên cạnh trong ngoặc vuông (VD: [Nghị định 82, Điều 5] hoặc [article_09.md]).
2. KHÔNG BỊA ĐẶT: Chỉ sử dụng thông tin CÓ TRONG tài liệu được cung cấp (context). Nếu tài liệu không đủ thông tin để trả lời trọn vẹn câu hỏi, hãy nói rõ: 'Tôi không thể xác minh thông tin này từ nguồn hiện có'.
3. TỔNG HỢP MẠCH LẠC: KHÔNG liệt kê hay chắp vá các đoạn văn một cách rời rạc. Hãy xâu chuỗi các ý thành một bài viết logic, có lời dẫn dắt, có sự liên kết chặt chẽ (sử dụng các từ nối như 'Theo đó', 'Tuy nhiên', 'Ngoài ra'). Văn phong phải mượt mà như một người đang tư vấn.
4. TRÌNH BÀY DỄ HIỂU: Cấu trúc rõ ràng, chia đoạn hợp lý, dùng in đậm để làm nổi bật các ý chính."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Split into first half (important → đầu) and second half (important → cuối)
    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])  # Odd positions go first
    for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
        reordered.append(chunks[i])  # Even positions go last (reversed)

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    # Step 5: Call LLM
    from openai import OpenAI
    
    # Hỗ trợ cấu hình API key và Base URL cho dịch vụ bên thứ 3 (như DeepSeek, Groq, local LM Studio...)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") # Nếu dùng OpenAI chuẩn thì cứ để trống trong .env
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), # Có thể cấu hình model trong .env luôn, mặc định là gpt-4o-mini
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        frequency_penalty=0.5, # Thêm tham số này để chống lỗi LLM bị lặp từ/lặp câu
        presence_penalty=0.5
    )

    answer = response.choices[0].message.content

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
