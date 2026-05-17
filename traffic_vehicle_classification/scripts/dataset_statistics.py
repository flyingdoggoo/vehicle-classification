from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

from common import get_label_names, iter_image_files, load_config, relative_to_project, resolve_project_path


def collect_image_rows(input_root, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    for label, path in tqdm(list(iter_image_files(input_root, labels)), desc="Reading image metadata"):
        try:
            with Image.open(path) as image:
                width, height = image.size
                rows.append(
                    {
                        "label": label,
                        "path": relative_to_project(path),
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height if height else 0,
                        "mode": image.mode,
                        "format": image.format,
                        "file_size_kb": round(path.stat().st_size / 1024, 2),
                    }
                )
        except Exception as exc:
            errors.append({"label": label, "path": relative_to_project(path), "error": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(errors)


def save_plots(df: pd.DataFrame, figures_dir) -> None:
    figures_dir = resolve_project_path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    sns.set_theme(style="whitegrid")
    order = df["label"].value_counts().index

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="label", order=order)
    plt.xticks(rotation=35, ha="right")
    plt.title("Image Count by Label")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(df["width"], bins=40, kde=True)
    plt.title("Image Width Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "width_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(df["height"], bins=40, kde=True, color="#3c8d57")
    plt.title("Image Height Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "height_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="label", y="height", order=order)
    plt.xticks(rotation=35, ha="right")
    plt.title("Image Height by Label")
    plt.tight_layout()
    plt.savefig(figures_dir / "height_boxplot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="label", y="file_size_kb", order=order)
    plt.xticks(rotation=35, ha="right")
    plt.title("File Size by Label")
    plt.tight_layout()
    plt.savefig(figures_dir / "file_size_boxplot.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset summary CSV and figures.")
    parser.add_argument("--config", default="configs/labels.yaml")
    parser.add_argument("--input-root", default="data/cleaned")
    parser.add_argument("--summary-csv", default="reports/dataset_summary.csv")
    parser.add_argument("--details-csv", default="reports/image_details.csv")
    parser.add_argument("--errors-csv", default="reports/image_errors.csv")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = get_label_names(config)
    input_root = resolve_project_path(args.input_root)

    df, errors_df = collect_image_rows(input_root, labels)
    resolve_project_path(args.details_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolve_project_path(args.details_csv), index=False)
    errors_df.to_csv(resolve_project_path(args.errors_csv), index=False)

    if df.empty:
        summary = pd.DataFrame(
            columns=[
                "label",
                "count",
                "percentage",
                "width_mean",
                "width_min",
                "width_max",
                "height_mean",
                "height_min",
                "height_max",
                "aspect_ratio_mean",
                "file_size_kb_mean",
                "invalid_images",
            ]
        )
    else:
        total = len(df)
        summary = (
            df.groupby("label")
            .agg(
                count=("path", "count"),
                width_mean=("width", "mean"),
                width_min=("width", "min"),
                width_max=("width", "max"),
                height_mean=("height", "mean"),
                height_min=("height", "min"),
                height_max=("height", "max"),
                aspect_ratio_mean=("aspect_ratio", "mean"),
                file_size_kb_mean=("file_size_kb", "mean"),
            )
            .reset_index()
        )
        summary["percentage"] = (summary["count"] / total * 100).round(2)
        invalid_counts = errors_df.groupby("label").size().to_dict() if not errors_df.empty else {}
        summary["invalid_images"] = summary["label"].map(invalid_counts).fillna(0).astype(int)
        for col in ["width_mean", "height_mean", "aspect_ratio_mean", "file_size_kb_mean"]:
            summary[col] = summary[col].round(2)

    summary.to_csv(resolve_project_path(args.summary_csv), index=False)
    save_plots(df, args.figures_dir)
    print(f"Summary: {resolve_project_path(args.summary_csv)}")
    print(f"Details: {resolve_project_path(args.details_csv)}")
    print(f"Figures: {resolve_project_path(args.figures_dir)}")


if __name__ == "__main__":
    main()
