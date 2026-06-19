from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from model_utils import (
    DATA_DIR,
    MODELS_DIR,
    Timer,
    evaluate_model,
    load_image_datasets,
    save_metrics_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing saved model on the test split and save metrics artifacts."
    )
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "mobilenet_waste_classifier.keras"),
        help="Path to the saved Keras model.",
    )
    parser.add_argument(
        "--metadata",
        default=str(MODELS_DIR / "mobilenet_metadata.json"),
        help="Path to the matching metadata JSON file.",
    )
    parser.add_argument(
        "--train-dir",
        default=str(DATA_DIR / "train"),
        help="Train split path used only for class-name inference.",
    )
    parser.add_argument(
        "--validation-dir",
        default=str(DATA_DIR / "validation"),
        help="Validation split path.",
    )
    parser.add_argument(
        "--test-dir",
        default=str(DATA_DIR / "test"),
        help="Test split path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    metadata_path = Path(args.metadata)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    preprocessing = metadata.get("preprocessing", "")
    preprocess_fn = None
    if "mobilenet_v2.preprocess_input" in preprocessing:
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

    train_ds, validation_ds, test_ds, class_names = load_image_datasets(
        train_dir=Path(args.train_dir),
        validation_dir=Path(args.validation_dir),
        test_dir=Path(args.test_dir),
        image_size=tuple(metadata["image_size"]),
        batch_size=32,
        preprocess_fn=preprocess_fn,
    )
    del train_ds
    del validation_ds

    model = tf.keras.models.load_model(model_path)
    with Timer() as timer:
        test_loss, _ = model.evaluate(test_ds, verbose=0)
        metrics = evaluate_model(model, test_ds, class_names)

    save_metrics_artifacts(
        model_name=model_path.stem,
        metrics=metrics,
        class_names=class_names,
        history=None,
        training_time_seconds=timer.elapsed,
        test_loss=test_loss,
    )
    print(f"Saved evaluation artifacts for {model_path.stem}")


if __name__ == "__main__":
    main()
