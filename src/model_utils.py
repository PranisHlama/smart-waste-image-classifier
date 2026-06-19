from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrix"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32


def ensure_output_dirs() -> None:
    for directory in (MODELS_DIR, METRICS_DIR, CONFUSION_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def list_class_names(dataset_dir: Path) -> list[str]:
    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def build_dataset_from_directory(
    dataset_dir: Path,
    class_names: list[str],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    label_to_index = {label: index for index, label in enumerate(class_names)}
    image_paths: list[str] = []
    labels: list[int] = []

    for class_name in class_names:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file():
                image_paths.append(str(image_path))
                labels.append(label_to_index[class_name])

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((path_ds, label_ds))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=123, reshuffle_each_iteration=True)

    def load_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image.set_shape((image_size[0], image_size[1], 3))
        image = tf.cast(image, tf.float32)
        return image, label

    dataset = dataset.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size)


def load_image_datasets(
    train_dir: Path,
    validation_dir: Path,
    test_dir: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    preprocess_fn=None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    class_names = list_class_names(train_dir)
    train_ds = build_dataset_from_directory(
        dataset_dir=train_dir,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_ds = build_dataset_from_directory(
        dataset_dir=validation_dir,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    test_ds = build_dataset_from_directory(
        dataset_dir=test_dir,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    if preprocess_fn is not None:
        train_ds = train_ds.map(lambda images, labels: (preprocess_fn(images), labels))
        validation_ds = validation_ds.map(
            lambda images, labels: (preprocess_fn(images), labels)
        )
        test_ds = test_ds.map(lambda images, labels: (preprocess_fn(images), labels))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    validation_ds = validation_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, validation_ds, test_ds, class_names


def plot_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    epochs = range(1, len(history.history["accuracy"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def render_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = confusion.max() / 2 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            color = "white" if confusion[i, j] > threshold else "black"
            plt.text(j, i, str(confusion[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

    return {
        "accuracy": accuracy,
        "precision_weighted": float(weighted_precision),
        "recall_weighted": float(weighted_recall),
        "f1_weighted": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
    }


def save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_text_report(report: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for label, values in report.items():
            if isinstance(values, dict):
                file.write(f"{label}\n")
                for key, value in values.items():
                    file.write(f"  {key}: {value}\n")
            else:
                file.write(f"{label}: {values}\n")
            file.write("\n")


def save_metrics_artifacts(
    model_name: str,
    metrics: dict,
    class_names: list[str],
    history: tf.keras.callbacks.History | None,
    training_time_seconds: float,
    test_loss: float,
) -> None:
    ensure_output_dirs()

    summary = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
        "test_loss": float(test_loss),
        "training_time_seconds": float(training_time_seconds),
        "class_names": class_names,
    }

    save_json(summary, METRICS_DIR / f"{model_name}_summary.json")
    save_json(
        metrics["classification_report"],
        METRICS_DIR / f"{model_name}_classification_report.json",
    )
    save_text_report(
        metrics["classification_report"],
        METRICS_DIR / f"{model_name}_classification_report.txt",
    )

    confusion = np.array(metrics["confusion_matrix"], dtype=int)
    np.savetxt(
        CONFUSION_DIR / f"{model_name}_confusion_matrix.csv",
        confusion,
        delimiter=",",
        fmt="%d",
    )
    render_confusion_matrix(
        confusion,
        class_names,
        CONFUSION_DIR / f"{model_name}_confusion_matrix.png",
    )
    if history is not None:
        plot_training_history(history, PLOTS_DIR / f"{model_name}_training_history.png")


def save_model_metadata(
    model_name: str,
    class_names: list[str],
    image_size: tuple[int, int],
    preprocessing: str,
) -> None:
    metadata = {
        "class_names": class_names,
        "image_size": list(image_size),
        "preprocessing": preprocessing,
    }
    save_json(metadata, MODELS_DIR / f"{model_name}_metadata.json")


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
