# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [x] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [x] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

Sơ đồ dưới đây khớp đúng code nhóm. A/B test chỉ khác nhau **đúng một bước Rerank** (tham số `use_reranking`); phần truy hồi Hybrid (vector + BM25 + RRF) và sinh đáp án là chung cho cả 2 config.

```text
                ┌────────────────────────────┐
                │  Golden Dataset (22 Q&A)    │  golden_dataset.json
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │      eval_pipeline.py       │  chạy A/B, chấm điểm, xuất báo cáo
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  RAG Pipeline (rag_pipeline.py)
                │  tham số: use_reranking      │
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────────────────┐
                │  HYBRID RETRIEVAL (chung cho A & B)      │
                │   • Semantic Search — ChromaDB           │  Task 5
                │     (OpenAI text-embedding-3-small)      │
                │   • Lexical Search — BM25                │  Task 6
                │   • RRF Fusion (gộp 2 nhánh theo rank)    │  Task 7
                └──────────────┬─────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
   Config A   │                                  │   Config B
 (use_reranking=True)                     (use_reranking=False)
   ┌──────────▼───────────┐            ┌─────────▼────────────┐
   │  Rerank — MMR (Task 7)│            │  (bỏ qua bước rerank) │
   └──────────┬───────────┘            └─────────┬────────────┘
              └────────────────┬────────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  Fallback PageIndex          │  Task 8
                │  (chỉ khi best_score < 0.3)  │
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  LLM Generation — gpt-4o-mini│  Task 10 (trả lời có citation)
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  DeepEval — 4 metrics        │  Faithfulness · Answer Relevancy
                │                              │  Context Recall · Context Precision
                └──────────────┬─────────────┘
                               │
                ┌──────────────▼─────────────┐
                │  results.md                  │  bảng điểm + phân tích + đề xuất
                └────────────────────────────┘
```

---

## Phân Công Công Việc (Chốt Option 2: RAG Evaluation)

| Thành viên | MSSV | Nhiệm vụ chi tiết (Bám sát Yêu cầu chung & Yêu cầu 2) | Trạng thái |
|-----------|------|-------------------------------------------------------|------------|
| **Đạt** | 2A202600549 | **Data, Report & Diagram (No-code):**<br>- Tạo `golden_dataset.json` (15+ cặp Q&A).<br>- Viết báo cáo phân tích vào `results.md`.<br>- Vẽ Sơ đồ Kiến trúc Hệ thống (Architecture Diagram) chèn vào README. | ✅ DONE |
| **Giang** | 2A202600624 | **Pipeline Integration (Code):**<br>- Lấy code cá nhân (Retrieval, Generation) ghép thành 1 RAG pipeline hoàn chỉnh.<br>- Code pipeline phải nhận tham số để switch được 2 configs (VD: Bật/Tắt Reranking) phục vụ A/B Test.<br>- Đảm bảo code chạy được để Demo. | ✅ DONE |
| **Lợi** | 2A202600784 | **Evaluation Script & A/B Test (Code):**<br>- Viết logic bằng DeepEval/RAGAS trong `eval_pipeline.py`.<br>- Chạy so sánh A/B 2 cấu hình từ pipeline của Giang.<br>- Quản lý Github chung (gom code, push repo). | ✅ DONE |

---

## Hướng Dẫn Chạy

> Nhóm chọn **Yêu cầu 2 — RAG Evaluation Pipeline** (không có chatbot app riêng).

```bash
# 1. Vào thư mục group_project
cd group_project

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Tạo file .env trong group_project/ (embedding, generation và DeepEval đều dùng OpenAI)
#    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# 4. (Tùy chọn) Build lại index vào ChromaDB nếu chưa có thư mục data/chroma_db
python -m src.task4_chunking_indexing

# 5. Chạy đánh giá A/B + xuất báo cáo
python evaluation/eval_pipeline.py
#    -> Kết quả được ghi vào evaluation/results.md
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
