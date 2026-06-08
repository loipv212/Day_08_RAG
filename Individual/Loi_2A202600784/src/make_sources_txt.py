import json
from pathlib import Path

NEWS_DIR = Path("data/landing/news")
OUTPUT_FILE = NEWS_DIR / "sources.txt"

lines = []

for file in sorted(NEWS_DIR.glob("article_*.json")):
    data = json.loads(file.read_text(encoding="utf-8"))

    filename = file.name
    source = data.get("source", "")
    title = data.get("title", "")
    url = data.get("url", "")

    lines.append(f"{filename} | {source} | {title} | {url}")

OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved: {OUTPUT_FILE}")