"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _is_garbled_text(text: str, min_letters: int = 200) -> bool:
    """
    Phát hiện text bị mất khoảng trắng giữa các từ (vd. "CăncứLuậtTổchức..."
    thay vì "Căn cứ Luật Tổ chức..."). Một số PDF khiến MarkItDown chọn nhầm
    backend trích xuất (pdfplumber thay vì pdfminer) và dính chữ liền nhau.

    Văn bản tiếng Việt bình thường có tỉ lệ khoảng_trắng/ký_tự_chữ ~0.15-0.25
    (nhiều từ đơn âm tiết); text bị dính liền có tỉ lệ thấp hơn nhiều.
    """
    letters = sum(1 for c in text if c.isalpha())
    if letters < min_letters:
        return False
    return text.count(" ") / letters < 0.08


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            text = result.text_content

            if filepath.suffix.lower() == ".pdf" and _is_garbled_text(text):
                print("  ⚠ Text bị dính liền (thiếu khoảng trắng) — fallback sang pdfminer trực tiếp")
                from pdfminer.high_level import extract_text
                text = extract_text(str(filepath))

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
