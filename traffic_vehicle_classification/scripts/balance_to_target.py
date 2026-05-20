from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from common import (
    get_label_names,
    iter_image_files,
    load_config,
    relative_to_project,
    resolve_project_path,
    write_csv,
)


REPORT_FIELDS = [
    "label",
    "path",
    "action",
    "reason",
    "width",
    "height",
    "output_path",
]


@dataclass
class ImageMeta:
    label: str
    path: Path
    width: int | None
    height: int | None
    unreadable: bool

    @property
    def area(self) -> int:
        if self.width is None or self.height is None:
            return -1
        return self.width * self.height

    def too_small(self, min_width: int, min_height: int) -> bool:
        if self.width is None or self.height is None:
            return True
        return self.width < min_width or self.height < min_height


def read_image_meta(label: str, path: Path) -> ImageMeta:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            return ImageMeta(label=label, path=path, width=width, height=height, unreadable=False)
    except Exception:
        return ImageMeta(label=label, path=path, width=None, height=None, unreadable=True)


def move_file(path: Path, label: str, output_root: Path) -> str:
    dest_dir = output_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    idx = 1
    while dest.exists():
        dest = dest_dir / f"{path.stem}_{idx}{path.suffix}"
        idx += 1
    shutil.move(str(path), str(dest))
    return relative_to_project(dest)


def remove_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def process_label(
    label: str,
    input_root: Path,
    output_root: Path,
    min_keep: int,
    max_keep: int,
    min_width: int,
    min_height: int,
    action: str,
    rng: random.Random,
) -> list[dict]:
    rows: list[dict] = []
    files = [path for current_label, path in iter_image_files(input_root, [label]) if current_label == label]
    if not files:
        return rows

    metas = [read_image_meta(label, path) for path in tqdm(files, desc=f"Scanning {label}", leave=False)]
    current_count = len(metas)
    if current_count <= max_keep:
        print(f"[KEEP] {label}: {current_count} (<= {max_keep})")
        return rows

    remove_needed = current_count - max_keep
    low_quality = [m for m in metas if m.unreadable or m.too_small(min_width, min_height)]
    low_quality.sort(key=lambda item: (0 if item.unreadable else 1, item.area))

    selected: list[tuple[ImageMeta, str]] = []
    for meta in low_quality:
        if len(selected) >= remove_needed:
            break
        selected.append((meta, "unreadable" if meta.unreadable else f"too_small:{meta.width}x{meta.height}"))

    if len(selected) < remove_needed:
        selected_paths = {item.path for item, _ in selected}
        rest = [m for m in metas if m.path not in selected_paths]
        rng.shuffle(rest)
        for meta in rest[: remove_needed - len(selected)]:
            selected.append((meta, "over_target_random_trim"))

    # Safe guard: never trim below min_keep.
    max_can_remove = max(0, current_count - min_keep)
    if len(selected) > max_can_remove:
        selected = selected[:max_can_remove]

    print(
        f"[TRIM] {label}: current={current_count}, remove={len(selected)}, "
        f"target_range=[{min_keep}, {max_keep}] -> expected={current_count - len(selected)}"
    )

    for meta, reason in selected:
        output_path = ""
        if action == "move":
            output_path = move_file(meta.path, label, output_root)
        else:
            remove_file(meta.path)
        rows.append(
            {
                "label": label,
                "path": relative_to_project(meta.path),
                "action": action,
                "reason": reason,
                "width": meta.width if meta.width is not None else "",
                "height": meta.height if meta.height is not None else "",
                "output_path": output_path,
            }
        )
    return rows


def drop_labels(labels_to_drop: list[str], input_root: Path, output_root: Path, action: str) -> list[dict]:
    rows: list[dict] = []
    for label in labels_to_drop:
        label_dir = input_root / label
        if not label_dir.exists():
            continue
        files = [path for path in label_dir.rglob("*") if path.is_file()]
        print(f"[DROP_LABEL] {label}: {len(files)} files")
        for path in tqdm(files, desc=f"Dropping {label}", leave=False):
            output_path = ""
            if action == "move":
                output_path = move_file(path, label, output_root / "dropped_labels")
            else:
                remove_file(path)
            rows.append(
                {
                    "label": label,
                    "path": relative_to_project(path),
                    "action": action,
                    "reason": "drop_label",
                    "width": "",
                    "height": "",
                    "output_path": output_path,
                }
            )
        if action == "delete":
            shutil.rmtree(label_dir, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Balance cleaned dataset into a target range. "
            "If a class is over max_keep, remove unreadable/small images first, "
            "then random trim until max_keep. Never trim below min_keep."
        )
    )
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/cleaned")
    parser.add_argument("--output-root", default="data/rejected/balance_to_target")
    parser.add_argument("--min-keep", type=int, default=1100)
    parser.add_argument("--max-keep", type=int, default=1200)
    parser.add_argument("--min-width", type=int, default=None)
    parser.add_argument("--min-height", type=int, default=None)
    parser.add_argument("--drop-labels", default="taxi")
    parser.add_argument("--action", choices=["move", "delete"], default="move")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="reports/balance_to_target_report.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    input_root = resolve_project_path(args.input_root)
    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cleaning_cfg = config.get("image_cleaning", {})
    min_width = int(args.min_width if args.min_width is not None else cleaning_cfg.get("min_width", 128))
    min_height = int(args.min_height if args.min_height is not None else cleaning_cfg.get("min_height", 128))
    rng = random.Random(args.seed)

    dropped = [item.strip() for item in args.drop_labels.split(",") if item.strip()]
    rows: list[dict] = []

    if dropped:
        rows.extend(drop_labels(dropped, input_root, output_root, args.action))
        labels = [label for label in labels if label not in set(dropped)]

    for label in labels:
        rows.extend(
            process_label(
                label=label,
                input_root=input_root,
                output_root=output_root,
                min_keep=args.min_keep,
                max_keep=args.max_keep,
                min_width=min_width,
                min_height=min_height,
                action=args.action,
                rng=rng,
            )
        )

    write_csv(args.report, rows, REPORT_FIELDS)
    print(f"Done. Actions: {len(rows)}")
    print(f"Report: {resolve_project_path(args.report)}")


if __name__ == "__main__":
    main()

