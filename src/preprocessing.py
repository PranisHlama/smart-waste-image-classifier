from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_as_array(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load one image, resize it, and normalize pixel values to 0-1."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_dataset_as_arrays(
    dataset_dir: Path, target_size: tuple[int, int] = (224, 224)
) -> tuple[np.ndarray, np.ndarray]:
    """Load a class-folder image dataset into image and label arrays."""
    images = []
    labels = []

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())

    for class_dir in class_dirs:
        label = class_dir.name

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            images.append(load_image_as_array(image_path, target_size))
            labels.append(label)

    return np.array(images), np.array(labels)


def create_augmented_images(image: Image.Image) -> dict[str, Image.Image]:
    """Create deterministic augmentations for a single image."""
    image = image.convert("RGB")

    return {
        "flip_horizontal": ImageOps.mirror(image),
        "rotate_left": image.rotate(15, resample=Image.Resampling.BICUBIC, expand=False),
        "rotate_right": image.rotate(-15, resample=Image.Resampling.BICUBIC, expand=False),
        "bright": ImageEnhance.Brightness(image).enhance(1.2),
        "contrast": ImageEnhance.Contrast(image).enhance(1.2),
    }
