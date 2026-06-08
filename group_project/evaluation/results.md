# RAG Evaluation Results

## Framework sử dụng
> DeepEval (với gpt-4o-mini)

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Không Rerank) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.94 | 0.82 | +0.12 |
| Answer Relevancy | 0.80 | 0.71 | +0.09 |
| Contextual Recall | 0.61 | 0.50 | +0.11 |
| Contextual Precision| 0.60 | 0.35 | +0.25 |
| **Average** | **0.74** | **0.59** | **+0.15** |

---

## A/B Comparison Analysis

**Config A:**
> Sử dụng Hybrid Search + Reranking (BAAI/bge-reranker-v2-m3) kết hợp LLM thế hệ mới.

**Config B:**
> Chỉ sử dụng Semantic Search cơ bản (Không có Reranking).

**Kết luận:**
> Config A tốt hơn rõ rệt (trung bình cao hơn 15%). Bộ Reranker đóng vai trò then chốt trong việc đẩy các tài liệu quan trọng lên đầu (Context Precision tăng vọt từ 0.35 lên 0.60). Nhờ có tài liệu đúng, AI tự tin trả lời chính xác hơn, dẫn đến điểm trung thực (Faithfulness) và trọng tâm (Relevancy) đều tăng mạnh.

---

## Worst Performers (Bottom 3)

1. **Câu hỏi:** "Ca sĩ Sơn Tùng M-TP có liên quan đến ma túy không?"
   * **Lý do sai:** Lỗi Context Precision = 0.0. Hệ thống tìm kiếm lấy nhầm thông tin của "Sơn Ngọc Minh" và "Miu Lê", dẫn đến AI bối rối.
2. **Câu hỏi:** "Quy trình cai nghiện bắt buộc kéo dài bao lâu?"
   * **Lý do sai:** Lỗi Context Recall thấp. Văn bản chia làm nhiều phần, hệ thống chỉ lấy được giai đoạn 1 mà thiếu giai đoạn 2.
3. **Câu hỏi:** "Tội tàng trữ ma túy bị phạt như thế nào?"
   * **Lý do sai:** Trả lời thiếu các tình tiết tăng nặng (khối lượng, tái phạm nguy hiểm) do chunk size bị cắt ngang.
