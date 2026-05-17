from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from common import PROJECT_ROOT, get_label_names, iter_image_files, load_config, resolve_project_path, write_csv


SUMMARY_FIELDS = ["label", "split", "count", "percentage_within_label"]


def clear_split_output(output_root: Path) -> None:
    output_root = output_root.resolve()
    project_root = PROJECT_ROOT.resolve()
    if output_root != project_root and project_root not in output_root.parents:
        raise ValueError(f"Refuse to clear path outside project: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n == 0:
        return 0, 0, 0
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
    while n_train + n_val + n_test > n:
        n_train = max(0, n_train - 1)
    while n_train + n_val + n_test < n:
        n_train += 1
    return n_train, n_val, n_test


def copy_or_move(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test folders.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/cleaned")
    parser.add_argument("--output-root", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["copy", "move"], default="copy")
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--summary", default="reports/split_summary.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    split_cfg = config.get("split", {})
    train_ratio = args.train_ratio if args.train_ratio is not None else float(split_cfg.get("train_ratio", 0.70))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(split_cfg.get("val_ratio", 0.15))
    test_ratio = args.test_ratio if args.test_ratio is not None else float(split_cfg.get("test_ratio", 0.15))
    seed = args.seed if args.seed is not None else int(split_cfg.get("random_seed", 42))

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    input_root = resolve_project_path(args.input_root)
    output_root = resolve_project_path(args.output_root)
    if args.clear_output:
        clear_split_output(output_root)

    for split in ["train", "val", "test"]:
        for label in labels:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    summary_rows = []

    for label in labels:
        files = [path for file_label, path in iter_image_files(input_root, [label]) if file_label == label]
        rng.shuffle(files)
        n_train, n_val, n_test = split_counts(len(files), train_ratio, val_ratio)
        split_map = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val : n_train + n_val + n_test],
        }

        for split, split_files in split_map.items():
            for src in split_files:
                copy_or_move(src, output_root / split / label / src.name, args.mode)
            percentage = round(len(split_files) / len(files) * 100, 2) if files else 0
            summary_rows.append(
                {
                    "label": label,
                    "split": split,
                    "count": len(split_files),
                    "percentage_within_label": percentage,
                }
            )

    write_csv(args.summary, summary_rows, SUMMARY_FIELDS)
    print(f"Split folders: {output_root}")
    print(f"Summary: {resolve_project_path(args.summary)}")


if __name__ == "__main__":
    main()
