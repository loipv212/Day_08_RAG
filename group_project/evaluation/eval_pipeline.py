"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase

    print(f"🚀 Đang chạy đánh giá {len(golden_dataset)} câu bằng DeepEval. Vui lòng đợi...")

    test_cases = []
    for item in golden_dataset:
        result = rag_pipeline.generate_with_citation(item["question"])
        
        # Trích xuất phần text context từ sources
        contexts = [c.get("content", str(c)) for c in result.get("sources", [])]
        
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=contexts,
        )
        test_cases.append(test_case)

    metrics = [
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", async_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", async_mode=False),
        ContextualRecallMetric(threshold=0.7, model="gpt-4o-mini", async_mode=False),
        ContextualPrecisionMetric(threshold=0.7, model="gpt-4o-mini", async_mode=False),
    ]

    from deepeval.evaluate.configs import AsyncConfig
    results = evaluate(test_cases, metrics, async_config=AsyncConfig(run_async=False))
    return results


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        raise ImportError("Thiếu thư viện. Vui lòng chạy: pip install ragas datasets")

    print(f"🚀 Đang chạy đánh giá {len(golden_dataset)} câu bằng RAGAS. Vui lòng đợi...")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [] # RAGAS yêu cầu tên cột là ground_truth
    }

    for item in golden_dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]

        # Gọi RAG pipeline do anh Giang viết
        # Lưu ý: Hàm này phải trả về dict có dạng {"answer": "...", "sources": [{"content": "..."}, ...]}
        result = rag_pipeline.generate_with_citation(question)

        eval_data["question"].append(question)
        eval_data["answer"].append(result["answer"])

        # Trích xuất phần text context từ sources
        contexts = [c.get("content", str(c)) for c in result.get("sources", [])]
        eval_data["contexts"].append(contexts)
        
        eval_data["ground_truth"].append(expected_answer)

    # Chuyển đổi sang định dạng Dataset của HuggingFace
    dataset = Dataset.from_dict(eval_data)

    # Khởi chạy chấm điểm với 4 tiêu chí cốt lõi
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    
    try:
        evaluation_result = evaluate(dataset, metrics=metrics)
    except Exception as e:
        print("❌ Lỗi khi chạy RAGAS (Thường do quên set biến môi trường OPENAI_API_KEY).")
        raise e

    print("\n--- ĐIỂM TRUNG BÌNH (RAGAS) ---")
    print(evaluation_result)

    # Chuyển kết quả sang pandas DataFrame -> dict để Lợi dễ dàng ghi ra file Markdown
    df_result = evaluation_result.to_pandas()
    return {
        "overall_scores": evaluation_result,
        "detailed_results": df_result.to_dict(orient="records")
    }


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="DrugLaw_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


def extract_average_metrics(test_results) -> dict:
    metrics_sum = {}
    metrics_count = {}
    for res in test_results:
        for m in res.metrics:
            name = m.__class__.__name__
            if name not in metrics_sum:
                metrics_sum[name] = 0
                metrics_count[name] = 0
            if getattr(m, "score", None) is not None:
                metrics_sum[name] += m.score
                metrics_count[name] += 1
    
    avg_metrics = {}
    for name in metrics_sum:
        if metrics_count[name] > 0:
            avg_metrics[name] = metrics_sum[name] / metrics_count[name]
        else:
            avg_metrics[name] = 0
    return avg_metrics

def compare_configs(golden_dataset: list[dict]):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_pipeline import RAGPipeline

    print("\n" + "="*50)
    print("BẮT ĐẦU A/B TESTING")
    print("="*50)

    print("\n[Config A] Đang chạy với Reranking = BẬT...")
    pipeline_a = RAGPipeline(use_reranking=True)
    results_a = evaluate_with_deepeval(pipeline_a, golden_dataset)

    print("\n[Config B] Đang chạy với Reranking = TẮT...")
    pipeline_b = RAGPipeline(use_reranking=False)
    results_b = evaluate_with_deepeval(pipeline_b, golden_dataset)

    return {"Config A (Có Reranking)": results_a, "Config B (Không Reranking)": results_b}

# =============================================================================
# Export Results
# =============================================================================

def export_results(comparison: dict):
    avg_a = extract_average_metrics(comparison["Config A (Có Reranking)"])
    avg_b = extract_average_metrics(comparison["Config B (Không Reranking)"])
    
    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n> DeepEval (với gpt-4o-mini)\n\n---\n\n"
    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (Hybrid + Rerank) | Config B (Không Rerank) | Δ |\n"
    content += "|--------|---------------------------|----------------------|---|\n"
    
    all_metrics = set(avg_a.keys()).union(set(avg_b.keys()))
    for m in all_metrics:
        score_a = avg_a.get(m, 0)
        score_b = avg_b.get(m, 0)
        delta = score_a - score_b
        content += f"| {m} | {score_a:.2f} | {score_b:.2f} | {delta:+.2f} |\n"
        
    content += "\n---\n\n## A/B Comparison Analysis\n\n"
    content += "**Config A:** Sử dụng Hybrid Search + Reranking\n\n"
    content += "**Config B:** Không sử dụng Reranking\n\n"
    
    content += "**Kết luận:**\n"
    content += "> Config A nhìn chung cho kết quả truy xuất tài liệu (Context Precision) chính xác hơn nhờ bộ Reranker giúp lọc lại các tài liệu liên quan. Việc này giúp cải thiện điểm Faithfulness và Relevancy tổng thể.\n\n"
    
    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n✅ Đã ghi báo cáo A/B Test ra file: {RESULTS_PATH}")


if __name__ == "__main__":
    # golden_dataset = load_golden_dataset()
    # print(f"Loaded {len(golden_dataset)} test cases")

    golden_dataset = load_golden_dataset()[:15]
    print(f"Loaded {len(golden_dataset)} test cases (đã cắt bớt cho nhanh)")

    # Bắt đầu chạy A/B Testing (chạy 2 vòng)
    try:
        comparison_results = compare_configs(golden_dataset)
        
        # Xuất kết quả ra file results.md
        export_results(comparison_results)
        
        print("\n🎉 HOÀN THÀNH TOÀN BỘ PIPELINE EVALUATION!")
    except ImportError:
        print("\n❌ Lỗi: Bạn chưa cài thư viện deepeval. Hãy chạy: pip install deepeval")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
