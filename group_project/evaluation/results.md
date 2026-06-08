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

**Config A (Hybrid + Rerank):**
> Hybrid Search (ChromaDB vector + BM25 lexical → RRF Fusion) **có** thêm bước Rerank bằng MMR (Task 7), rồi sinh đáp án bằng gpt-4o-mini (`use_reranking=True`).

**Config B (Hybrid, không Rerank):**
> Cùng luồng Hybrid Search (vector + BM25 + RRF) như Config A nhưng **bỏ** bước Rerank (`use_reranking=False`). A/B test chỉ khác nhau đúng bước rerank này.

**Kết luận:**
> Config A tốt hơn rõ rệt (trung bình cao hơn 15%). Bộ Reranker đóng vai trò then chốt trong việc đẩy các tài liệu quan trọng lên đầu (Context Precision tăng vọt từ 0.35 lên 0.60). Nhờ có tài liệu đúng, AI tự tin trả lời chính xác hơn, dẫn đến điểm trung thực (Faithfulness) và trọng tâm (Relevancy) đều tăng mạnh.

---

## Worst Performers (Bottom 3)

1. **Câu hỏi:** "Ca sĩ Sơn Tùng M-TP có bị khởi tố vì liên quan đến ma túy không?"
   * **Lý do sai:** Lỗi Context Precision = 0.0. Đây là câu out-of-scope; hệ thống tìm kiếm lấy nhầm thông tin của "Sơn Ngọc Minh" và "Miu Lê", dẫn đến AI bối rối.
2. **Câu hỏi:** "Quy trình cai nghiện ma túy theo Luật Phòng, chống ma túy 2021 gồm những giai đoạn nào?"
   * **Lý do sai:** Lỗi Context Recall thấp. Văn bản (Điều 29) chia làm nhiều giai đoạn, hệ thống chỉ lấy được phần đầu mà thiếu các giai đoạn sau.
3. **Câu hỏi:** "Theo Điều 249 Bộ luật Hình sự, người phạm tội tàng trữ trái phép chất ma túy ở khung cơ bản (khoản 1) bị phạt tù bao nhiêu năm?"
   * **Lý do sai:** Trả lời thiếu các tình tiết tăng nặng (khối lượng, tái phạm nguy hiểm) do chunk size bị cắt ngang.

---

## Đề xuất Cải tiến (Recommendations)

Bám theo 3 worst performers và chênh lệch A/B ở trên, nhóm đề xuất các hướng cải thiện cụ thể:

### 1. Khắc phục retrieval lấy nhầm ngữ cảnh (Context Precision thấp)
- **Vấn đề:** câu out-of-scope ("Sơn Tùng M-TP") kéo về nhầm chunk của "Sơn Ngọc Minh" / "Miu Lê" vì cùng ngữ cảnh "ca sĩ + ma túy".
- **Đề xuất:**
  - Nâng `SCORE_THRESHOLD` (Task 9) để loại chunk điểm thấp trước khi đưa vào LLM.
  - Thêm bước đối chiếu thực thể (tên riêng) giữa câu hỏi và chunk trước khi sinh đáp án.
  - Giữ **Reranking (Config A)** làm mặc định — đã đẩy Context Precision từ 0.35 → 0.60.

### 2. Tăng Context Recall cho câu hỏi nhiều phần
- **Vấn đề:** "quy trình cai nghiện" trải trên nhiều điều/đoạn, hệ thống chỉ lấy được một phần (thiếu giai đoạn 2).
- **Đề xuất:**
  - Tăng `top_k` khi truy hồi (vd 5 → 8) rồi để reranker lọc lại.
  - Tăng chunk overlap (Task 4) để không cắt ngang các giai đoạn liền nhau.
  - Cân nhắc section-aware / parent-document chunking: giữ trọn một Điều luật trong một chunk.

### 3. Tránh cắt cụt tình tiết pháp lý (chunk size)
- **Vấn đề:** câu tội tàng trữ trả lời thiếu tình tiết tăng nặng do chunk bị cắt ngang theo số ký tự.
- **Đề xuất:** tăng `CHUNK_SIZE` cho văn bản luật, hoặc chunk theo cấu trúc Điều/Khoản thay vì cắt cố định theo độ dài.

### Tổng kết
- **Chọn Config A (Hybrid + Reranking) làm cấu hình chính thức** vì vượt Config B +0.15 điểm trung bình.
- Nút thắt cần ưu tiên là **Context Recall (0.61)** và **Context Precision (0.60)** — hai điểm thấp nhất, kéo theo Faithfulness và Answer Relevancy.
