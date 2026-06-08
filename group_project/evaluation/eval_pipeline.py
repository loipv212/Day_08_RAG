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
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


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


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    # TODO: Implement A/B comparison
    #
    # configs = {
    #     "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
    #     "dense_only": {"use_reranking": False, "alpha": 1.0},
    # }
    #
    # results = {}
    # for config_name, params in configs.items():
    #     # Run eval with this config
    #     ...
    #     results[config_name] = scores
    #
    # return results
    raise NotImplementedError("Implement compare_configs")


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    # TODO: Format and write results
    #
    # content = "# RAG Evaluation Results\n\n"
    # content += "## Overall Scores\n\n"
    # content += "| Metric | Score |\n|--------|-------|\n"
    # ...
    # content += "\n## A/B Comparison\n\n"
    # ...
    # content += "\n## Worst Performers\n\n"
    # ...
    # content += "\n## Recommendations\n\n"
    # ...
    #
    # RESULTS_PATH.write_text(content, encoding="utf-8")
    raise NotImplementedError("Implement export_results")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # Import RAG pipeline từ gốc dự án nhóm (đã được Giang đóng gói)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rag_pipeline import pipeline_default

    print("\n[TEST] Bắt đầu chạy Evaluation Pipeline với RAGAS...")
    
    # Lợi có thể comment lại các dòng dưới nếu chưa muốn chạy ngay
    # results = evaluate_with_ragas(pipeline_default, golden_dataset)
    
    print("\n✅ Tích hợp Pipeline thành công! Lợi đã có thể bắt đầu code tiếp phần A/B Testing và Export Results.")
