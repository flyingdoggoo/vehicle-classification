from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from common import get_label_names, iter_image_files, load_config, relative_to_project, resolve_project_path, write_csv


REPORT_FIELDS = ["label", "source_path", "output_path", "status", "reason", "width", "height"]


def handle_bad_file(path: Path, label: str, bad_root: Path, policy: str) -> str:
    if policy == "none":
        return ""
    if policy == "delete":
        path.unlink(missing_ok=True)
        return ""

    dest_dir = bad_root / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1

    if policy == "move":
        shutil.move(str(path), str(dest))
    else:
        shutil.copy2(path, dest)
    return relative_to_project(dest)


def clean_image(src_path: Path, dest_path: Path, min_width: int, min_height: int, jpeg_quality: int):
    try:
        with Image.open(src_path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width < min_width or height < min_height:
                return "rejected", f"too_small:{width}x{height}", width, height

            image = image.convert("RGB")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest_path, format="JPEG", quality=jpeg_quality, optimize=True)
            return "cleaned", "", width, height
    except Exception as exc:
        return "rejected", f"unreadable:{exc}", "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, normalize, and rename raw images.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/raw")
    parser.add_argument("--output-root", default="data/cleaned")
    parser.add_argument("--bad-root", default="data/rejected")
    parser.add_argument("--bad-policy", choices=["none", "copy", "move", "delete"], default="copy")
    parser.add_argument("--report", default="reports/cleaning_report.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    cleaning_cfg = config.get("image_cleaning", {})
    min_width = int(cleaning_cfg.get("min_width", 128))
    min_height = int(cleaning_cfg.get("min_height", 128))
    jpeg_quality = int(cleaning_cfg.get("jpeg_quality", 95))

    input_root = resolve_project_path(args.input_root)
    output_root = resolve_project_path(args.output_root)
    bad_root = resolve_project_path(args.bad_root)
    rows: list[dict] = []
    counters = {label: 1 for label in labels}

    for label in labels:
        (output_root / label).mkdir(parents=True, exist_ok=True)
        (bad_root / label).mkdir(parents=True, exist_ok=True)

    files = list(iter_image_files(input_root, labels))
    for label, src_path in tqdm(files, desc="Cleaning images"):
        dest_path = output_root / label / f"{label}_{counters[label]:06d}.jpg"
        while dest_path.exists() and not args.overwrite:
            counters[label] += 1
            dest_path = output_root / label / f"{label}_{counters[label]:06d}.jpg"

        status, reason, width, height = clean_image(src_path, dest_path, min_width, min_height, jpeg_quality)
        if status == "cleaned":
            output_path = relative_to_project(dest_path)
            counters[label] += 1
        else:
            output_path = handle_bad_file(src_path, label, bad_root, args.bad_policy)

        rows.append(
            {
                "label": label,
                "source_path": relative_to_project(src_path),
                "output_path": output_path,
                "status": status,
                "reason": reason,
                "width": width,
                "height": height,
            }
        )

    write_csv(args.report, rows, REPORT_FIELDS)
    print(f"Cleaned: {sum(row['status'] == 'cleaned' for row in rows)}")
    print(f"Rejected: {sum(row['status'] != 'cleaned' for row in rows)}")
    print(f"Report: {resolve_project_path(args.report)}")


if __name__ == "__main__":
    main()
