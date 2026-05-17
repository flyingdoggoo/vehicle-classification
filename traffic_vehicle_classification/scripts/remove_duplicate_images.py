from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from common import get_label_names, iter_image_files, load_config, relative_to_project, resolve_project_path, write_csv


REPORT_FIELDS = [
    "label",
    "duplicate_path",
    "kept_label",
    "kept_path",
    "hash_method",
    "hash",
    "kept_hash",
    "hamming_distance",
    "action",
]


def compute_hash(path: Path, method: str):
    with Image.open(path) as image:
        if method == "phash":
            return imagehash.phash(image)
        if method == "dhash":
            return imagehash.dhash(image)
        if method == "ahash":
            return imagehash.average_hash(image)
    raise ValueError(f"Unsupported hash method: {method}")


def place_duplicate(path: Path, label: str, duplicate_root: Path, action: str) -> str:
    if action == "delete":
        path.unlink(missing_ok=True)
        return ""

    dest_dir = duplicate_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1

    if action == "move":
        shutil.move(str(path), str(dest))
    else:
        shutil.copy2(path, dest)
    return relative_to_project(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove near-duplicate images using perceptual hash.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/cleaned")
    parser.add_argument("--duplicates-root", default="data/duplicates")
    parser.add_argument("--report", default="reports/duplicates.csv")
    parser.add_argument("--method", choices=["phash", "dhash", "ahash"], default=None)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--action", choices=["move", "copy", "delete"], default="move")
    parser.add_argument("--within-label-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    dedup_cfg = config.get("deduplication", {})
    method = args.method or dedup_cfg.get("hash_method", "phash")
    threshold = args.threshold if args.threshold is not None else int(dedup_cfg.get("hamming_threshold", 6))
    input_root = resolve_project_path(args.input_root)
    duplicates_root = resolve_project_path(args.duplicates_root)

    kept: list[tuple[str, Path, object]] = []
    rows: list[dict] = []

    for label, path in tqdm(list(iter_image_files(input_root, labels)), desc=f"Deduplicating ({method})"):
        try:
            current_hash = compute_hash(path, method)
        except Exception as exc:
            print(f"[WARN] Cannot hash {path}: {exc}")
            continue

        duplicate_match = None
        duplicate_distance = None
        for kept_label, kept_path, kept_hash in kept:
            if args.within_label_only and kept_label != label:
                continue
            distance = current_hash - kept_hash
            if distance <= threshold:
                duplicate_match = (kept_label, kept_path, kept_hash)
                duplicate_distance = distance
                break

        if duplicate_match is None:
            kept.append((label, path, current_hash))
            continue

        kept_label, kept_path, kept_hash = duplicate_match
        duplicate_output = place_duplicate(path, label, duplicates_root, args.action)
        rows.append(
            {
                "label": label,
                "duplicate_path": duplicate_output or relative_to_project(path),
                "kept_label": kept_label,
                "kept_path": relative_to_project(kept_path),
                "hash_method": method,
                "hash": str(current_hash),
                "kept_hash": str(kept_hash),
                "hamming_distance": duplicate_distance,
                "action": args.action,
            }
        )

    write_csv(args.report, rows, REPORT_FIELDS)
    print(f"Kept images: {len(kept)}")
    print(f"Duplicates handled: {len(rows)}")
    print(f"Report: {resolve_project_path(args.report)}")


if __name__ == "__main__":
    main()
