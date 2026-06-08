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


def _pdf_text_fallback(filepath: Path) -> str:
    """Trích text từ PDF bằng pypdf (dự phòng khi MarkItDown ra rỗng)."""
    from pypdf import PdfReader

    reader = PdfReader(str(filepath))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt.strip())
    return "\n\n".join(parts)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        print(f"Converting: {filepath.name}")

        text = ""
        try:
            text = (md.convert(str(filepath)).text_content or "").strip()
        except Exception as e:
            print(f"  ⚠ MarkItDown lỗi ({e}), thử pypdf...")

        # Fallback: PDF mà MarkItDown ra rỗng → dùng pypdf
        if not text and filepath.suffix.lower() == ".pdf":
            text = _pdf_text_fallback(filepath).strip()

        if not text:
            print(f"  ✗ Bỏ qua (không trích được nội dung): {filepath.name}")
            continue

        header = f"# {filepath.stem}\n\n**Nguồn:** {filepath.name} (legal)\n\n---\n\n"
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + text, encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name} ({len(text)} chars)")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))

        # Thêm metadata header để giữ thông tin nguồn (phục vụ citation ở Task 10)
        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

        content = header + data.get("content_markdown", "")
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name} ({len(content)} chars)")


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
