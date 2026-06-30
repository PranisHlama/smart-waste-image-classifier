from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_AUGMENTED_DIR, TRAIN_DIR
from preprocessing import IMAGE_EXTENSIONS, create_augmented_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create augmented copies for images in class-folder datasets."
    )
    parser.add_argument(
        "--source",
        default=str(TRAIN_DIR),
        help="Class-folder dataset to augment. Usually data/train.",
    )
    parser.add_argument(
        "--output",
        default=str(TRAIN_AUGMENTED_DIR),
        help="Directory where original and augmented images are saved.",
    )
    parser.add_argument(
        "--include-originals",
        action="store_true",
        help="Copy original images into the output directory with augmented images.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_augmented_images(
    image_path: Path,
    output_class_dir: Path,
    include_original: bool,
) -> int:
    output_class_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        if include_original:
            image.save(output_class_dir / image_path.name)
            saved_count += 1

        for augmentation_name, augmented_image in create_augmented_images(image).items():
            output_path = output_class_dir / f"{image_path.stem}_{augmentation_name}.jpg"
            augmented_image.save(output_path, quality=95)
            saved_count += 1

    return saved_count


def augment_dataset(
    source_dir: Path,
    output_dir: Path,
    include_originals: bool = False,
) -> int:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    class_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {source_dir}")

    total_saved = 0
    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            print(f"Skipping empty class folder: {class_dir.name}")
            continue

        class_saved = 0
        output_class_dir = output_dir / class_dir.name
        for image_path in images:
            class_saved += save_augmented_images(
                image_path=image_path,
                output_class_dir=output_class_dir,
                include_original=include_originals,
            )

        total_saved += class_saved
        print(f"{class_dir.name}: saved {class_saved} images")

    return total_saved


def main() -> None:
    args = parse_args()
    total_saved = augment_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        include_originals=args.include_originals,
    )
    print(f"Done: saved {total_saved} images")


if __name__ == "__main__":
    main()
