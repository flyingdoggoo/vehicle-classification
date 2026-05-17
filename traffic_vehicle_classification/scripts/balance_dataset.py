from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import get_label_names, iter_image_files, load_config, relative_to_project, resolve_project_path, write_csv


REPORT_FIELDS = ["label", "path", "action", "output_path", "reason"]


def place_file(path: Path, label: str, output_root: Path, action: str) -> str:
    if action == "delete":
        path.unlink(missing_ok=True)
        return ""

    dest_dir = output_root / label
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
        path.unlink(missing_ok=True)
    return relative_to_project(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Limit each class folder to a balanced maximum number of images.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/cleaned")
    parser.add_argument("--output-root", default="data/rejected/over_target")
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--action", choices=["delete", "move"], default="delete")
    parser.add_argument("--report", default="reports/balance_report.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    max_per_label = args.max_per_label or int(config.get("target_raw_per_label", 1500))
    seed = args.seed if args.seed is not None else int(config.get("split", {}).get("random_seed", 42))
    input_root = resolve_project_path(args.input_root)
    output_root = resolve_project_path(args.output_root)

    rows: list[dict] = []
    rng = random.Random(seed)

    for label in labels:
        paths = [path for current_label, path in iter_image_files(input_root, [label]) if current_label == label]
        paths = sorted(paths)
        if len(paths) <= max_per_label:
            print(f"[OK] {label}: {len(paths)} <= {max_per_label}")
            continue

        rng.shuffle(paths)
        keep_set = set(paths[:max_per_label])
        remove_paths = sorted([path for path in paths if path not in keep_set])
        print(f"[BALANCE] {label}: keep={max_per_label}, remove={len(remove_paths)}")

        for path in remove_paths:
            output_path = ""
            if not args.dry_run:
                output_path = place_file(path, label, output_root, args.action)
            rows.append(
                {
                    "label": label,
                    "path": relative_to_project(path),
                    "action": "dry_run" if args.dry_run else args.action,
                    "output_path": output_path,
                    "reason": f"over_max_per_label:{max_per_label}",
                }
            )

    write_csv(args.report, rows, REPORT_FIELDS)
    print(f"Balanced labels to max_per_label={max_per_label}")
    print(f"Handled files: {len(rows)}")
    print(f"Report: {resolve_project_path(args.report)}")


if __name__ == "__main__":
    main()
