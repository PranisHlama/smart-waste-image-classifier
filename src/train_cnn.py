from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

from model_utils import (
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    MODELS_DIR,
    Timer,
    ensure_output_dirs,
    evaluate_model,
    load_image_datasets,
    save_metrics_artifacts,
    save_model_metadata,
)


MODEL_NAME = "cnn_waste_classifier"
TRAIN_DIR = DATA_DIR / "train_augmented"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
EPOCHS = 10


def build_model(num_classes: int, image_size: tuple[int, int]) -> tf.keras.Model:
    return models.Sequential(
        [
            layers.Input(shape=(image_size[0], image_size[1], 3)),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def main() -> None:
    ensure_output_dirs()

    train_ds, validation_ds, test_ds, class_names = load_image_datasets(
        train_dir=TRAIN_DIR,
        validation_dir=VALIDATION_DIR,
        test_dir=TEST_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    model = build_model(len(class_names), DEFAULT_IMAGE_SIZE)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    with Timer() as timer:
        history = model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
        )

    test_loss, _ = model.evaluate(test_ds, verbose=0)
    metrics = evaluate_model(model, test_ds, class_names)

    model.save(MODELS_DIR / f"{MODEL_NAME}.keras")
    save_model_metadata(
        model_name=MODEL_NAME,
        class_names=class_names,
        image_size=DEFAULT_IMAGE_SIZE,
        preprocessing="layers.Rescaling(1.0 / 255.0)",
    )
    save_metrics_artifacts(
        model_name=MODEL_NAME,
        metrics=metrics,
        class_names=class_names,
        history=history,
        training_time_seconds=timer.elapsed,
        test_loss=test_loss,
    )

    print(f"Saved model: {MODELS_DIR / f'{MODEL_NAME}.keras'}")


if __name__ == "__main__":
    main()
