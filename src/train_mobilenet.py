from __future__ import annotations

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


MODEL_NAME = "mobilenet_waste_classifier"
TRAIN_DIR = DATA_DIR / "train_augmented"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
EPOCHS = 5


def build_model(num_classes: int) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    ensure_output_dirs()

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    train_ds, validation_ds, test_ds, class_names = load_image_datasets(
        train_dir=TRAIN_DIR,
        validation_dir=VALIDATION_DIR,
        test_dir=TEST_DIR,
        image_size=DEFAULT_IMAGE_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        preprocess_fn=preprocess_input,
    )

    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
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
        preprocessing="tf.keras.applications.mobilenet_v2.preprocess_input",
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
