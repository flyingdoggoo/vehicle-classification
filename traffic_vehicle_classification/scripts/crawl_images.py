from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

from common import (
    IMAGE_EXTENSIONS,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
    append_csv,
    count_images_by_label,
    ensure_label_dirs,
    get_label_items,
    get_label_names,
    load_config,
    next_index_for_label,
    relative_to_project,
    resolve_project_path,
    slugify,
)


METADATA_FIELDS = [
    "label",
    "keyword",
    "source",
    "path",
    "original_path",
    "image_url",
    "page_url",
    "title",
    "width",
    "height",
    "downloaded_at",
]


def parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def as_bool_text(value: bool) -> str:
    return "true" if value else "false"


def image_size(path: Path) -> tuple[int | str, int | str]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return "", ""


def count_images_in_label_dir(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    return sum(1 for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def unique_raw_path(raw_root: Path, label: str, source: str, keyword: str, suffix: str = ".jpg") -> Path:
    index = next_index_for_label(raw_root, label)
    label_dir = raw_root / label
    keyword_slug = slugify(keyword)
    source_slug = slugify(source)
    suffix = suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"

    while True:
        candidate = label_dir / f"{label}_{index:06d}_{source_slug}_{keyword_slug}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_autocrawler_command(
    autocrawler_path: Path,
    config: dict,
    sources: set[str],
    threads_override: int | None,
    limit_override: int | None,
) -> list[str]:
    autocrawler_cfg = config.get("autocrawler", {})
    threads = threads_override or int(autocrawler_cfg.get("threads", 4))
    full_resolution = bool(autocrawler_cfg.get("full_resolution", False))
    no_gui = str(autocrawler_cfg.get("no_gui", "auto"))
    limit_per_keyword = limit_override or int(autocrawler_cfg.get("limit_per_keyword", 120))

    return [
        sys.executable,
        str(autocrawler_path / "main.py"),
        "--skip",
        "false",
        "--threads",
        str(threads),
        "--google",
        as_bool_text("google" in sources),
        "--naver",
        as_bool_text("naver" in sources),
        "--full",
        as_bool_text(full_resolution),
        "--no_gui",
        no_gui,
        "--limit",
        str(limit_per_keyword),
    ]


def copy_autocrawler_outputs(run_dir: Path, raw_root: Path, label: str, target_count: int) -> list[dict]:
    rows: list[dict] = []
    download_dir = run_dir / "download"
    if not download_dir.exists():
        return rows

    for keyword_dir in sorted(path for path in download_dir.iterdir() if path.is_dir()):
        keyword = keyword_dir.name
        for src_path in sorted(keyword_dir.iterdir()):
            if src_path.name.endswith("_done") or not src_path.is_file():
                continue
            if src_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if count_images_in_label_dir(raw_root / label) >= target_count:
                return rows

            source = src_path.stem.split("_", 1)[0]
            if source not in {"google", "naver"}:
                source = "autocrawler"

            dest_path = unique_raw_path(raw_root, label, source, keyword, src_path.suffix)
            shutil.copy2(src_path, dest_path)
            width, height = image_size(dest_path)
            rows.append(
                {
                    "label": label,
                    "keyword": keyword,
                    "source": source,
                    "path": relative_to_project(dest_path),
                    "original_path": str(src_path),
                    "image_url": "",
                    "page_url": "",
                    "title": "",
                    "width": width,
                    "height": height,
                    "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return rows


def run_autocrawler_for_label(
    label_item: dict,
    config: dict,
    sources: set[str],
    raw_root: Path,
    target_count: int,
    autocrawler_path: Path,
    dry_run: bool,
    threads_override: int | None,
    limit_override: int | None,
) -> list[dict]:
    enabled_sources = {"google", "naver"} & sources
    if not enabled_sources:
        return []

    label = label_item["name"]
    keywords = label_item["keywords"]
    run_dir = PROJECT_ROOT / "data" / "crawl_runs" / label / f"autocrawler_{time.strftime('%Y%m%d_%H%M%S')}"
    command = build_autocrawler_command(
        autocrawler_path=autocrawler_path,
        config=config,
        sources=enabled_sources,
        threads_override=threads_override,
        limit_override=limit_override,
    )

    if dry_run:
        print(f"[DRY RUN] AutoCrawler label={label}")
        print(f"  keywords: {', '.join(keywords)}")
        print(f"  cwd: {run_dir}")
        print(f"  command: {' '.join(command)}")
        return []

    if not (autocrawler_path / "main.py").exists():
        raise FileNotFoundError(f"AutoCrawler main.py not found: {autocrawler_path}")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "keywords.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(keywords) + "\n")

    print(f"[AutoCrawler] {label}: crawling from {', '.join(sorted(enabled_sources))}")
    result = subprocess.run(
        command,
        cwd=run_dir,
        input="n\n",
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"[WARN] AutoCrawler returned code {result.returncode} for {label}")

    return copy_autocrawler_outputs(run_dir, raw_root, label, target_count)


def download_ddg_image(image_url: str, dest_path: Path, timeout: int) -> tuple[int, int]:
    import requests
    from PIL import Image, ImageOps

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        )
    }
    response = requests.get(image_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image = ImageOps.exif_transpose(image).convert("RGB")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path, format="JPEG", quality=95, optimize=True)
    return image.size


def run_duckduckgo_for_label(
    label_item: dict,
    config: dict,
    raw_root: Path,
    target_count: int,
    dry_run: bool,
) -> list[dict]:
    label = label_item["name"]
    if dry_run:
        print(f"[DRY RUN] DuckDuckGo fallback label={label}")
        return []

    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ImportError("Install DuckDuckGo support first: pip install ddgs") from exc

    cfg = config.get("duckduckgo", {})
    timeout = int(cfg.get("timeout_seconds", 15))
    rows: list[dict] = []

    for keyword in label_item["keywords"]:
        remaining = target_count - count_images_in_label_dir(raw_root / label)
        if remaining <= 0:
            break

        print(f"[DuckDuckGo] {label}: keyword='{keyword}', remaining={remaining}")
        try:
            results = DDGS(timeout=timeout).images(
                query=keyword,
                region=cfg.get("region", "us-en"),
                safesearch=cfg.get("safesearch", "moderate"),
                max_results=min(int(cfg.get("max_results_per_keyword", 180)), max(remaining * 2, 25)),
                size=cfg.get("size", None),
            )
        except Exception as exc:
            print(f"[WARN] DuckDuckGo search failed for '{keyword}': {exc}")
            continue

        for result in results:
            if count_images_in_label_dir(raw_root / label) >= target_count:
                break

            image_url = result.get("image") or result.get("thumbnail")
            if not image_url:
                continue

            dest_path = unique_raw_path(raw_root, label, "duckduckgo", keyword, ".jpg")
            try:
                width, height = download_ddg_image(image_url, dest_path, timeout)
            except Exception as exc:
                print(f"[WARN] Failed to download DDG image for {label}: {exc}")
                continue

            rows.append(
                {
                    "label": label,
                    "keyword": keyword,
                    "source": "duckduckgo",
                    "path": relative_to_project(dest_path),
                    "original_path": "",
                    "image_url": image_url,
                    "page_url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "width": width,
                    "height": height,
                    "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl vehicle images into data/raw/<label>.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--labels", default=None, help="Comma-separated labels, e.g. car,bus,truck.")
    parser.add_argument("--sources", default=None, help="Comma-separated sources: google,naver,duckduckgo.")
    parser.add_argument("--limit-per-label", type=int, default=None)
    parser.add_argument("--autocrawler-limit-per-keyword", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--metadata", default="reports/crawl_metadata.csv")
    parser.add_argument("--autocrawler-path", default=str(WORKSPACE_ROOT / "autocrawler"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    label_items = get_label_items(config, parse_csv_arg(args.labels))
    raw_root = resolve_project_path(args.raw_root)
    ensure_label_dirs(raw_root, labels)

    selected_sources = set(parse_csv_arg(args.sources) or config.get("sources", ["google", "naver", "duckduckgo"]))
    invalid_sources = selected_sources - {"google", "naver", "duckduckgo"}
    if invalid_sources:
        raise ValueError(f"Unsupported sources: {', '.join(sorted(invalid_sources))}")

    target_count = args.limit_per_label or int(config.get("target_raw_per_label", 1500))
    autocrawler_path = Path(args.autocrawler_path).resolve()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Selected sources: {', '.join(sorted(selected_sources))}")
    print(f"Target raw images per label: {target_count}")

    counts = count_images_by_label(raw_root, labels)
    for label_item in label_items:
        label = label_item["name"]
        if counts.get(label, 0) >= target_count:
            print(f"[SKIP] {label}: already has {counts[label]} images")
            continue

        print(f"[START] {label}: current={counts.get(label, 0)}, target={target_count}")
        rows = run_autocrawler_for_label(
            label_item=label_item,
            config=config,
            sources=selected_sources,
            raw_root=raw_root,
            target_count=target_count,
            autocrawler_path=autocrawler_path,
            dry_run=args.dry_run,
            threads_override=args.threads,
            limit_override=args.autocrawler_limit_per_keyword,
        )
        append_csv(args.metadata, rows, METADATA_FIELDS)

        if "duckduckgo" in selected_sources and count_images_in_label_dir(raw_root / label) < target_count:
            rows = run_duckduckgo_for_label(label_item, config, raw_root, target_count, args.dry_run)
            append_csv(args.metadata, rows, METADATA_FIELDS)

        print(f"[DONE] {label}: raw_count={count_images_in_label_dir(raw_root / label)}")


if __name__ == "__main__":
    main()
