"""
RAG Pipeline Interface (Dành cho Group Project).
Đã được cấu hình để chạy độc lập trong thư mục group_project.
"""

from src.task10_generation import generate_with_citation as real_generate

class RAGPipeline:
    def __init__(self, use_reranking=True):
        """
        Khởi tạo Pipeline cho hệ thống RAG.
        :param use_reranking: Cho phép chạy A/B Testing bật/tắt reranking.
        """
        self.use_reranking = use_reranking

    def generate_with_citation(self, query: str) -> dict:
        """
        Hàm chính được sử dụng bởi Chatbot và Hệ thống Đánh giá.
        Trực tiếp gọi vào logic đã viết của Task 10.
        """
        return real_generate(query=query, top_k=5, use_reranking=self.use_reranking)

# Instance mặc định dùng chung
pipeline_default = RAGPipeline(use_reranking=True)
