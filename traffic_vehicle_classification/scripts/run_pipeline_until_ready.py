from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from common import IMAGE_EXTENSIONS, PROJECT_ROOT, get_label_items, get_label_names, load_config, resolve_project_path


PROGRESS_FIELDS = ["round", "label", "raw_count", "cleaned_count", "target", "missing", "timestamp"]


def parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def count_images(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    return sum(1 for path in label_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def count_by_label(root: Path, labels: list[str]) -> dict[str, int]:
    return {label: count_images(root / label) for label in labels}


def run_command(command: list[str], timeout: int | None = None) -> int:
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, timeout=timeout)
    if result.returncode != 0:
        print(f"[WARN] Command returned code {result.returncode}", flush=True)
    return result.returncode


def append_progress(path: Path, round_index: int, raw_counts: dict[str, int], clean_counts: dict[str, int], labels: list[str], target: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PROGRESS_FIELDS)
        if write_header:
            writer.writeheader()
        for label in labels:
            cleaned = clean_counts.get(label, 0)
            writer.writerow(
                {
                    "round": round_index,
                    "label": label,
                    "raw_count": raw_counts.get(label, 0),
                    "cleaned_count": cleaned,
                    "target": target,
                    "missing": max(0, target - cleaned),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )


def print_counts(title: str, raw_counts: dict[str, int], clean_counts: dict[str, int], labels: list[str], target: int) -> None:
    print(f"\n{title}")
    print("-" * 72)
    print(f"{'label':<14}{'raw':>8}{'cleaned':>10}{'missing':>10}{'status':>12}")
    for label in labels:
        cleaned = clean_counts.get(label, 0)
        missing = max(0, target - cleaned)
        status = "OK" if missing == 0 else "NEED_MORE"
        print(f"{label:<14}{raw_counts.get(label, 0):>8}{cleaned:>10}{missing:>10}{status:>12}")


def run_processing_steps(args) -> None:
    run_command([sys.executable, "scripts/remove_corrupted_images.py", "--overwrite"])
    run_command(
        [
            sys.executable,
            "scripts/remove_duplicate_images.py",
            "--method",
            args.hash_method,
            "--threshold",
            str(args.hamming_threshold),
            "--action",
            "move",
        ]
    )
    run_command([sys.executable, "scripts/dataset_statistics.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat crawl-clean-dedup-stat until each label has enough cleaned images.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--labels", default=None, help="Comma-separated labels. Default: all labels.")
    parser.add_argument("--sources", default=None, help="Comma-separated sources. Default uses sources from configs/labels.yaml.")
    parser.add_argument("--target-clean-per-label", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=300, help="Number of additional raw images to request per missing label each round.")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means keep going until done or stalled.")
    parser.add_argument("--max-stall-rounds", type=int, default=2)
    parser.add_argument("--hash-method", choices=["phash", "dhash", "ahash"], default=None)
    parser.add_argument("--hamming-threshold", type=int, default=None)
    parser.add_argument("--skip-final-split", action="store_true")
    parser.add_argument("--progress-csv", default="reports/pipeline_progress.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    all_labels = get_label_names(config)
    selected_items = get_label_items(config, parse_csv_arg(args.labels))
    labels = [item["name"] for item in selected_items]
    target = args.target_clean_per_label or int(config.get("target_raw_per_label", 1500))
    args.hash_method = args.hash_method or config.get("deduplication", {}).get("hash_method", "phash")
    args.hamming_threshold = args.hamming_threshold if args.hamming_threshold is not None else int(config.get("deduplication", {}).get("hamming_threshold", 6))

    raw_root = resolve_project_path("data/raw")
    cleaned_root = resolve_project_path("data/cleaned")
    progress_path = resolve_project_path(args.progress_csv)
    previous_total_cleaned = -1
    stall_rounds = 0
    round_index = 0

    print(f"Target cleaned images per label: {target}")
    print(f"Labels: {', '.join(labels)}")
    if args.sources is None:
        args.sources = ",".join(config.get("sources", ["google", "naver", "duckduckgo"]))

    print(f"Sources: {args.sources}")
    print(f"Batch size: {args.batch_size}")

    while True:
        round_index += 1
        raw_counts = count_by_label(raw_root, all_labels)
        clean_counts = count_by_label(cleaned_root, all_labels)
        print_counts(f"Before round {round_index}", raw_counts, clean_counts, labels, target)
        append_progress(progress_path, round_index, raw_counts, clean_counts, labels, target)

        missing_labels = [label for label in labels if clean_counts.get(label, 0) < target]
        if not missing_labels:
            print("\nAll selected labels reached the target.")
            break

        if args.max_rounds and round_index > args.max_rounds:
            print("\nReached max rounds. You can run this script again to resume.")
            break

        total_cleaned = sum(clean_counts.get(label, 0) for label in labels)
        if total_cleaned == previous_total_cleaned:
            stall_rounds += 1
        else:
            stall_rounds = 0
        previous_total_cleaned = total_cleaned
        if stall_rounds > args.max_stall_rounds:
            print("\nNo cleaned-image progress for several rounds. Stop to avoid an infinite loop.")
            print("Try more keywords, another source, or inspect crawl logs.")
            break

        for label in missing_labels:
            current_raw = raw_counts.get(label, 0)
            raw_goal = current_raw + max(1, args.batch_size)
            print(f"\n[Crawl] label={label}, current_raw={current_raw}, next_raw_goal={raw_goal}")
            run_command(
                [
                    sys.executable,
                    "scripts/crawl_images.py",
                    "--labels",
                    label,
                    "--sources",
                    args.sources,
                    "--limit-per-label",
                    str(raw_goal),
                ]
            )

        run_processing_steps(args)

    run_processing_steps(args)
    if not args.skip_final_split:
        run_command([sys.executable, "scripts/split_dataset.py", "--clear-output"])

    raw_counts = count_by_label(raw_root, all_labels)
    clean_counts = count_by_label(cleaned_root, all_labels)
    print_counts("Final counts", raw_counts, clean_counts, labels, target)
    append_progress(progress_path, round_index, raw_counts, clean_counts, labels, target)
    print(f"\nProgress log: {progress_path}")


if __name__ == "__main__":
    main()
