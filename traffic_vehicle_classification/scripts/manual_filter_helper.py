from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
from pathlib import Path

from common import get_label_names, iter_image_files, load_config, resolve_project_path, write_csv


MANIFEST_FIELDS = ["label", "path", "action", "target_label", "note"]
VALID_ACTIONS = {"", "keep", "reject", "review", "move"}


def project_relative(path: Path) -> str:
    try:
        return path.relative_to(resolve_project_path(".")).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(input_root: Path, labels: list[str], selected_label: str | None, max_per_label: int) -> list[dict]:
    rows: list[dict] = []
    counts = {label: 0 for label in labels}
    selected = {selected_label} if selected_label else set(labels)

    for label, path in iter_image_files(input_root, labels):
        if label not in selected or counts[label] >= max_per_label:
            continue
        counts[label] += 1
        rows.append(
            {
                "label": label,
                "path": project_relative(path),
                "action": "",
                "target_label": "",
                "note": "",
            }
        )
    return rows


def write_html(rows: list[dict], output_html: Path, thumb_size: int) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        image_path = resolve_project_path(row["path"])
        rel = html.escape(os.path.relpath(image_path, output_html.parent).replace("\\", "/"), quote=True)
        label = html.escape(row["label"])
        path_text = html.escape(row["path"])
        cards.append(
            f"""
            <article class="card">
              <img src="{rel}" alt="{label}" loading="lazy">
              <div class="meta">
                <strong>{label}</strong>
                <code>{path_text}</code>
              </div>
            </article>
            """
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Vehicle Image Review</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f7f7f5;
      color: #1f2933;
    }}
    header {{
      position: sticky;
      top: 0;
      background: #ffffff;
      border-bottom: 1px solid #d9dee3;
      padding: 14px 20px;
      z-index: 1;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax({thumb_size + 34}px, 1fr));
      gap: 14px;
      padding: 18px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d9dee3;
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      width: 100%;
      height: {thumb_size}px;
      object-fit: cover;
      display: block;
      background: #e8ecef;
    }}
    .meta {{
      padding: 10px;
      display: grid;
      gap: 6px;
      font-size: 13px;
    }}
    code {{
      font-size: 11px;
      overflow-wrap: anywhere;
      color: #5c6773;
    }}
  </style>
</head>
<body>
  <header>
    <strong>Manual Vehicle Image Review</strong>
    <span> Edit reports/manual_review_manifest.csv, then run --apply-actions.</span>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    output_html.write_text(content, encoding="utf-8")


def unique_dest(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
        counter += 1
    return dest


def apply_actions(actions_csv: Path, rejected_root: Path, review_root: Path, cleaned_root: Path) -> None:
    with open(actions_csv, "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    moved = 0
    for row in rows:
        action = (row.get("action") or "").strip().lower()
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}' for {row.get('path')}")
        if action in {"", "keep"}:
            continue

        src = resolve_project_path(row["path"])
        if not src.exists():
            print(f"[WARN] Missing file, skip: {src}")
            continue

        label = row["label"]
        if action == "reject":
            dest = unique_dest(rejected_root / label, src.name)
        elif action == "review":
            dest = unique_dest(review_root / label, src.name)
        else:
            target_label = (row.get("target_label") or "").strip()
            if not target_label:
                raise ValueError(f"target_label is required for move action: {row.get('path')}")
            dest = unique_dest(cleaned_root / target_label, src.name)

        shutil.move(str(src), str(dest))
        moved += 1
    print(f"Applied actions. Moved files: {moved}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a review grid and apply manual filtering actions.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/cleaned")
    parser.add_argument("--output-html", default="reports/manual_review.html")
    parser.add_argument("--manifest", default="reports/manual_review_manifest.csv")
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-per-label", type=int, default=120)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--apply-actions", default=None)
    parser.add_argument("--rejected-root", default="data/rejected")
    parser.add_argument("--review-root", default="data/manual_review")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    input_root = resolve_project_path(args.input_root)

    if args.apply_actions:
        apply_actions(
            actions_csv=resolve_project_path(args.apply_actions),
            rejected_root=resolve_project_path(args.rejected_root),
            review_root=resolve_project_path(args.review_root),
            cleaned_root=input_root,
        )
        return

    rows = build_manifest(input_root, labels, args.label, args.max_per_label)
    write_csv(args.manifest, rows, MANIFEST_FIELDS)
    write_html(rows, resolve_project_path(args.output_html), args.thumb_size)
    print(f"Manifest: {resolve_project_path(args.manifest)}")
    print(f"HTML review grid: {resolve_project_path(args.output_html)}")


if __name__ == "__main__":
    main()
