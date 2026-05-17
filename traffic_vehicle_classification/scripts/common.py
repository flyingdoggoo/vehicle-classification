from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "labels.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(resolve_project_path(path), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_label_names(config: dict) -> list[str]:
    return [item["name"] for item in config["labels"]]


def get_label_items(config: dict, selected: Iterable[str] | None = None) -> list[dict]:
    labels = config["labels"]
    if selected is None:
        return labels

    selected_set = {item.strip() for item in selected if item.strip()}
    known = {item["name"] for item in labels}
    missing = sorted(selected_set - known)
    if missing:
        raise ValueError(f"Unknown labels: {', '.join(missing)}")

    return [item for item in labels if item["name"] in selected_set]


def ensure_label_dirs(root: str | Path, labels: Iterable[str]) -> None:
    root = resolve_project_path(root)
    root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (root / label).mkdir(parents=True, exist_ok=True)


def iter_image_files(root: str | Path, labels: Iterable[str] | None = None):
    root = resolve_project_path(root)
    if labels is None:
        label_dirs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    else:
        label_dirs = [root / label for label in labels]

    for label_dir in label_dirs:
        if not label_dir.exists():
            continue
        label = label_dir.name
        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label, path


def count_images_by_label(root: str | Path, labels: Iterable[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for label, _ in iter_image_files(root, labels):
        counts[label] = counts.get(label, 0) + 1
    return counts


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "keyword"


def next_index_for_label(root: str | Path, label: str) -> int:
    label_dir = resolve_project_path(root) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(label)}_(\d+)")
    max_index = 0
    for path in label_dir.iterdir():
        match = pattern.match(path.stem)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def append_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    if not rows:
        return

    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_to_project(path: str | Path) -> str:
    path = resolve_project_path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
