import argparse
import json
import os
import re
import sys
import time
from collections import deque
from typing import Optional

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "WikipediaCrawler/1.0 (assignment corpus builder; contact: student)"
}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Wikipedia_Entries")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _api_get(params: dict, retries: int = 6) -> Optional[dict]:

    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5 * (2**attempt)))
                wait = max(retry_after, 5 * (2**attempt))
                print(
                    f"  [rate-limit] 429 on attempt {attempt + 1}, waiting {wait}s …",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if attempt == retries - 1:
                print(
                    f"  [warn] API error ({exc}) for params {params}", file=sys.stderr
                )
                return None
            time.sleep(5 * (2**attempt))
        except Exception as exc:
            if attempt == retries - 1:
                print(
                    f"  [warn] API error ({exc}) for params {params}", file=sys.stderr
                )
                return None
            time.sleep(3 * (attempt + 1))
    return None


def fetch_page_text(title: str) -> Optional[str]:
    data = _api_get(
        {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "format": "json",
            "formatversion": "2",
        }
    )
    if data is None:
        return None
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    return pages[0].get("extract", "").strip() or None


def fetch_page_links(title: str, limit: int = 30) -> list[str]:
    links = []
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "pllimit": min(limit, 500),
        "plnamespace": 0,
        "format": "json",
        "formatversion": "2",
    }
    data = _api_get(params)
    if data is None:
        return links
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return links
    for lnk in pages[0].get("links", []):
        links.append(lnk["title"])
    return links[:limit]


def _safe_filename(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", title)


def save_page(title: str, text: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    doc = {"id": title, "title": title, "text": text}
    fname = _safe_filename(title) + ".json"
    with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    os.makedirs(output_dir, exist_ok=True)
    doc = {"id": title, "title": title, "text": text}
    fname = _safe_filename(title) + ".json"
    with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def already_saved(title: str, output_dir: str) -> bool:
    fname = _safe_filename(title) + ".json"
    return os.path.exists(os.path.join(output_dir, fname))


def crawl(
    seed: str, max_depth: int, max_pages: int, links_per_page: int, output_dir: str
) -> None:
    queue: deque[tuple[str, int]] = deque()  # (title, depth)
    visited: set[str] = set()

    queue.append((seed, 0))
    visited.add(seed.lower())

    saved = 0
    pbar = tqdm(total=max_pages, desc="Pages saved", unit="page")

    while queue and saved < max_pages:
        title, depth = queue.popleft()

        # ---- Fetch text ----
        if not already_saved(title, output_dir):
            text = fetch_page_text(title)
            time.sleep(0.35)  # polite delay – ~3 req/s well under rate limit
            if text is None:
                continue
            save_page(title, text, output_dir)
        else:
            text = "cached"  # file already exists; still enqueue links

        saved += 1
        pbar.update(1)
        pbar.set_postfix({"depth": depth, "queue": len(queue), "title": title[:30]})

        # ---- Enqueue links ----
        if depth < max_depth and saved < max_pages:
            links = fetch_page_links(title, limit=links_per_page)
            time.sleep(0.35)
            for link in links:
                if link.lower() not in visited:
                    visited.add(link.lower())
                    queue.append((link, depth + 1))

    pbar.close()
    print(f"\nDone. {saved} pages saved to {os.path.abspath(output_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate data/Wikipedia_Entries/ via BFS crawl."
    )
    parser.add_argument(
        "--seed", default="Cell (biology)", help="Starting Wikipedia page title"
    )
    parser.add_argument("--depth", type=int, default=3, help="BFS depth (default: 3)")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=300,
        help="Maximum pages to save (default: 300)",
    )
    parser.add_argument(
        "--links-per-page",
        type=int,
        default=20,
        help="Links to follow per page (default: 20)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, help="Output directory for JSON files"
    )
    args = parser.parse_args()

    print(f"Seed       : {args.seed}")
    print(f"Depth      : {args.depth}")
    print(f"Max pages  : {args.max_pages}")
    print(f"Links/page : {args.links_per_page}")
    print(f"Output dir : {os.path.abspath(args.output_dir)}")
    print()

    crawl(
        seed=args.seed,
        max_depth=args.depth,
        max_pages=args.max_pages,
        links_per_page=args.links_per_page,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
