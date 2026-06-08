"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Cách tiếp cận:
    Thay vì Crawl4AI (nặng, cần Playwright/Chromium — khó cài trên Windows + Python 3.14),
    ở đây dùng `requests` để tải HTML và `markitdown` (Microsoft) để chuyển sang Markdown.
    Nhẹ, không cần trình duyệt headless, đủ tốt cho các trang báo tĩnh.

Output:
    data/landing/news/article_XX.json — mỗi bài 1 file JSON gồm:
        { url, title, date_crawled, source_domain, content_markdown }

Chạy:
    python src/task2_crawl_news.py
"""

import io
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Giả lập trình duyệt để tránh bị chặn 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.9",
}

# Danh sách URL bài báo về nghệ sĩ Việt liên quan ma tuý
ARTICLE_URLS = [
    "https://www.nguoiduatin.vn/loat-sao-viet-vuong-vong-lao-ly-vi-ma-tuy-hao-quang-khong-phai-la-chan-cho-chat-cam-204260515121550023.htm",
    "https://baovanhoa.vn/giai-tri/ma-tuy-va-nhung-cu-nga-ngua-cua-showbiz-viet-230477.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://tienphong.vn/bi-hai-chuyen-nghe-si-test-ma-tuy-post1847129.tpo",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_title(html: str, fallback: str) -> str:
    """Lấy tiêu đề bài báo: ưu tiên og:title → <title> → fallback."""
    soup = BeautifulSoup(html, "html.parser")
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return fallback


def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo: tải HTML → convert markdown → trả về metadata + content.

    Returns:
        { url, title, date_crawled, source_domain, content_markdown }
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text

    title = extract_title(html, fallback=url)

    # MarkItDown convert HTML bytes → markdown
    md = MarkItDown()
    stream = io.BytesIO(html.encode("utf-8"))
    result = md.convert_stream(stream, file_extension=".html")
    content = (result.text_content or "").strip()
    # Gọn bớt dòng trống thừa
    content = re.sub(r"\n{3,}", "\n\n", content)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "source_domain": urlparse(url).netloc,
        "content_markdown": content,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS (bỏ trùng URL)."""
    setup_directory()
    seen = set()
    saved = 0

    for url in ARTICLE_URLS:
        if url in seen:
            print(f"  ⊘ Bỏ qua link trùng: {url}")
            continue
        seen.add(url)
        saved += 1
        print(f"[{saved}] Crawling: {url}")
        try:
            article = crawl_article(url)
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")
            saved -= 1
            continue

        filepath = DATA_DIR / f"article_{saved:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath.name} "
              f"({len(article['content_markdown'])} chars) — {article['title'][:60]}")

    print(f"\nTổng cộng: {saved} bài đã lưu vào {DATA_DIR}")
    if saved < 5:
        print(f"⚠ Mới có {saved} bài, cần ≥5. Thêm URL vào ARTICLE_URLS.")


if __name__ == "__main__":
    crawl_all()
