from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, IMAGE_DIR, IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split class-folder image dataset into train/validation/test folders."
    )
    parser.add_argument("--source", default=str(IMAGE_DIR), help="Source class-folder directory.")
    parser.add_argument("--output", default=str(DATA_DIR), help="Directory where split folders are created.")
    parser.add_argument("--train", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--validation", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing train/validation/test folders before splitting.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_split_dirs(output_dir: Path, overwrite: bool) -> None:
    for split_name in ("train", "validation", "test"):
        split_dir = output_dir / split_name
        if split_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{split_dir} already exists. Use --overwrite to recreate the split."
                )
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def copy_images(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        shutil.copy2(image_path, destination / image_path.name)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    split_total = args.train + args.validation + args.test
    if round(split_total, 6) != 1.0:
        raise ValueError("Split ratios must add up to 1.0")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {source_dir}")

    prepare_split_dirs(output_dir, args.overwrite)

    rng = random.Random(args.seed)
    totals = {"train": 0, "validation": 0, "test": 0}

    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            print(f"Skipping empty class folder: {class_dir.name}")
            continue

        rng.shuffle(images)

        train_count = int(len(images) * args.train)
        validation_count = int(len(images) * args.validation)

        split_images = {
            "train": images[:train_count],
            "validation": images[train_count : train_count + validation_count],
            "test": images[train_count + validation_count :],
        }

        for split_name, split_paths in split_images.items():
            copy_images(split_paths, output_dir / split_name / class_dir.name)
            totals[split_name] += len(split_paths)

        print(
            f"{class_dir.name}: "
            f"{len(split_images['train'])} train, "
            f"{len(split_images['validation'])} validation, "
            f"{len(split_images['test'])} test"
        )

    print(
        "Done: "
        f"{totals['train']} train, "
        f"{totals['validation']} validation, "
        f"{totals['test']} test"
    )


if __name__ == "__main__":
    main()
