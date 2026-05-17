from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import IMAGE_EXTENSIONS, get_label_names, load_config, relative_to_project, resolve_project_path, write_csv


REPORT_FIELDS = ["label", "path", "action", "reason"]


def collect_duplicate_files(root: Path, labels: list[str]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for label in labels:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append((label, path))
    return files


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only a small duplicate sample set for reporting and delete the rest.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--duplicates-root", default="data/duplicates")
    parser.add_argument("--keep-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report", default="reports/duplicates_cleanup_report.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    seed = args.seed if args.seed is not None else int(config.get("split", {}).get("random_seed", 42))
    duplicates_root = resolve_project_path(args.duplicates_root)

    files = collect_duplicate_files(duplicates_root, labels)
    rng = random.Random(seed)
    rng.shuffle(files)

    keep_total = max(0, args.keep_total)
    keep_set = {path for _, path in files[:keep_total]}
    rows: list[dict] = []

    for label, path in sorted(files, key=lambda item: str(item[1])):
        if path in keep_set:
            rows.append({"label": label, "path": relative_to_project(path), "action": "keep", "reason": "sample_for_report"})
            continue
        rows.append({"label": label, "path": relative_to_project(path), "action": "dry_run_delete" if args.dry_run else "delete", "reason": f"keep_total:{keep_total}"})
        if not args.dry_run:
            path.unlink(missing_ok=True)

    if not args.dry_run:
        remove_empty_dirs(duplicates_root)

    write_csv(args.report, rows, REPORT_FIELDS)
    print(f"Duplicate files before cleanup: {len(files)}")
    print(f"Kept duplicate samples: {min(keep_total, len(files))}")
    print(f"Deleted duplicate files: {max(0, len(files) - keep_total) if not args.dry_run else 0}")
    print(f"Report: {resolve_project_path(args.report)}")


if __name__ == "__main__":
    main()
