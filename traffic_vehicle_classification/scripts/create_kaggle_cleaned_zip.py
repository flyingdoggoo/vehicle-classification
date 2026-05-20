from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_NAMES = {".gitignore", ".gitkeep", "Thumbs.db", "desktop.ini"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Kaggle-friendly zip from data/cleaned using forward-slash paths."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data") / "cleaned",
        help="Folder with class subfolders. Default: data/cleaned",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("traffic_vehicle_cleaned_kaggle.zip"),
        help="Output zip path. Default: traffic_vehicle_cleaned_kaggle.zip",
    )
    parser.add_argument(
        "--root-name",
        default="cleaned",
        help="Top-level folder name inside the zip. Default: cleaned",
    )
    parser.add_argument(
        "--include-all-files",
        action="store_true",
        help="Include non-image files too. By default only image files are zipped.",
    )
    return parser.parse_args()


def should_include_file(path: Path, include_all_files: bool) -> bool:
    if path.name in EXCLUDED_NAMES:
        return False
    if any(part.startswith("__pycache__") for part in path.parts):
        return False
    if include_all_files:
        return True
    return path.suffix.lower() in IMAGE_EXTENSIONS


def create_zip(source: Path, output: Path, root_name: str, include_all_files: bool) -> Counter:
    source = source.resolve()
    output = output.resolve()

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file_path in sorted(source.rglob("*")):
            if not file_path.is_file() or not should_include_file(file_path, include_all_files):
                continue

            relative_path = file_path.relative_to(source)
            if len(relative_path.parts) < 2:
                continue

            label = relative_path.parts[0]
            archive_path = Path(root_name, *relative_path.parts).as_posix()
            zf.write(file_path, archive_path)
            counts[label] += 1

    return counts


def validate_zip(output: Path) -> None:
    with zipfile.ZipFile(output, "r") as zf:
        bad_names = [name for name in zf.namelist() if "\\" in name]
        hidden_names = [name for name in zf.namelist() if Path(name).name in EXCLUDED_NAMES]

    if bad_names:
        preview = "\n".join(bad_names[:10])
        raise RuntimeError(f"Zip still contains backslash paths:\n{preview}")

    if hidden_names:
        preview = "\n".join(hidden_names[:10])
        raise RuntimeError(f"Zip still contains excluded files:\n{preview}")


def main() -> None:
    args = parse_args()
    counts = create_zip(args.source, args.output, args.root_name, args.include_all_files)
    validate_zip(args.output)

    total = sum(counts.values())
    print(f"Created Kaggle zip: {args.output.resolve()}")
    print(f"Total image files: {total}")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
