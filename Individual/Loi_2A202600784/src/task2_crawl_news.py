"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://www.vietnamplus.vn/bo-cong-an-canh-bao-ve-loai-ma-tuy-cuc-doc-fentanyl-post959705.vnp",
    "https://www.vietnamplus.vn/canh-bao-ve-loai-ma-tuy-cuc-doc-fentanyl-doc-tinh-cao-gap-50-lan-so-voi-heroine-post1082175.vnp",
    "https://nhandan.vn/chong-toi-pham-ma-tuy-trong-tinh-hinh-moi-ky-1-post844311.html",
    "https://cand.com.vn/Hoat-dong-LL-CAND/tang-cuong-hop-tac-dau-tranh-toi-pham-ma-tuy-co-to-chuc-xuyen-quoc-gia-tren-toan-the-gioi--i707751/",
    "https://tuoitre.vn/ti-le-tai-nghien-kha-lon-can-quy-dinh-trach-nhiem-cua-gia-dinh-chinh-quyen-dia-phuong-20201113083933193.htm",
    "https://nhandan.vn/thanh-cong-cai-nghien-ma-tuy-tai-gia-dinh-va-cong-dong-post567028.html",
    "https://nhandan.vn/rao-can-cai-nghien-ma-tuy-tai-cong-dong-post576191.html",
    "https://cand.com.vn/van-de-hom-nay-thoi-su/quan-ly-phong-ngua-dau-vao-cua-toi-pham-ma-tuy-ky-4--i658538/",
    "https://cand.com.vn/su-kien-binh-luan-thoi-su/quy-dinh-chat-che-khong-de-nguoi-nghien-loi-dung-tron-tranh-cai-nghien-bat-buoc-i646556/",
    "https://thanhnien.vn/hiem-hoa-bong-cuoi-tags1160141.html",
    "https://nhandan.vn/ngan-chan-viec-mua-ban-su-dung-bong-cuoi-post362754.html",
    "https://bocongan.gov.vn/chinh-sach-phap-luat/chi-tiet-cau-hoi/f2dc8e83-cd88-4521-b295-2fc687fbf169",
]


def get_source_from_url(url: str) -> str:
    """Lấy tên domain nguồn báo từ URL."""
    return urlparse(url).netloc.replace("www.", "")


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "source": str,
            "date_crawled": str,
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        if not result.success:
            return {
                "url": url,
                "title": "",
                "source": get_source_from_url(url),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": "",
                "error": result.error_message,
            }

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title", "Unknown")

        return {
            "url": url,
            "title": title,
            "source": get_source_from_url(url),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename

        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if article.get("error"):
            print(f"  ✗ Crawl lỗi nhưng vẫn lưu log: {filepath}")
            print(f"  Error: {article['error']}")
        else:
            print(f"  ✓ Saved: {filepath}")

        await asyncio.sleep(2)


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())