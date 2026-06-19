from __future__ import annotations

import csv
import json
from pathlib import Path

from model_utils import DATA_DIR, METRICS_DIR, ensure_output_dirs


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("images", "train", "validation", "test", "train_augmented")


def count_split_images(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not split_dir.exists():
        return counts

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for image_path in class_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def build_summary() -> dict:
    split_counts = {split: count_split_images(DATA_DIR / split) for split in SPLITS}
    classes = sorted({label for counts in split_counts.values() for label in counts})

    per_class = []
    for label in classes:
        row = {"class_name": label}
        total = 0
        for split in SPLITS:
            count = split_counts[split].get(label, 0)
            row[split] = count
            total += count if split == "images" else 0
        per_class.append(row)

    totals = {
        split: sum(split_counts[split].values())
        for split in SPLITS
    }

    return {
        "classes": classes,
        "num_classes": len(classes),
        "totals": totals,
        "per_class": per_class,
    }


def save_csv(summary: dict, output_path: Path) -> None:
    fieldnames = ["class_name", *SPLITS]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["per_class"]:
            writer.writerow(row)


def main() -> None:
    ensure_output_dirs()
    summary = build_summary()

    json_path = METRICS_DIR / "dataset_summary.json"
    csv_path = METRICS_DIR / "dataset_summary.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_csv(summary, csv_path)
    print(f"Saved dataset summary to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
