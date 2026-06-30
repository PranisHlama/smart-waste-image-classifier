from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from config import (
    CNN_LABELS_PATH,
    CNN_MODEL_PATH,
    IMAGE_SIZE,
    MOBILENET_LABELS_PATH,
    MOBILENET_MODEL_PATH,
)


MODEL_OPTIONS = {
    "MobileNetV2": {
        "model_path": MOBILENET_MODEL_PATH,
        "labels_path": MOBILENET_LABELS_PATH,
        "train_command": "python models/mobilenet.py",
    },
    "Custom CNN": {
        "model_path": CNN_MODEL_PATH,
        "labels_path": CNN_LABELS_PATH,
        "train_command": "python models/cnn.py",
    },
}


def read_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        return []
    return [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def predict(image: Image.Image, model: tf.keras.Model, labels: list[str]) -> list[tuple[str, float]]:
    predictions = model.predict(prepare_image(image), verbose=0)[0]
    predictions = np.asarray(predictions, dtype=np.float32)
    if np.isclose(np.sum(predictions), 1.0, atol=1e-3) and np.all(predictions >= 0):
        probabilities = predictions
    else:
        probabilities = tf.nn.softmax(predictions).numpy()

    if len(labels) != len(probabilities):
        labels = [f"Class {index}" for index in range(len(probabilities))]

    ranked_indexes = np.argsort(probabilities)[::-1]
    return [(labels[index], float(probabilities[index])) for index in ranked_indexes]


def show_setup_message(model_name: str, model_path: Path, labels_path: Path, train_command: str) -> None:
    st.warning(f"{model_name} is not ready yet.")
    st.write("Train the model first, then restart the app.")
    st.code(train_command, language="bash")
    st.caption(f"Expected model: {model_path}")
    st.caption(f"Expected labels: {labels_path}")


def main() -> None:
    st.set_page_config(page_title="Smart Waste Classifier", layout="centered")
    st.title("Smart Waste Image Classifier")

    model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[model_name]
    model_path = selected_model["model_path"]
    labels_path = selected_model["labels_path"]

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if not model_path.exists() or not labels_path.exists():
        show_setup_message(
            model_name=model_name,
            model_path=model_path,
            labels_path=labels_path,
            train_command=selected_model["train_command"],
        )
        return

    labels = read_labels(labels_path)
    model = load_model(str(model_path))

    if uploaded_file is None:
        st.info("Upload an image to classify it.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    ranked_predictions = predict(image, model, labels)
    top_label, top_confidence = ranked_predictions[0]

    st.subheader(top_label)
    st.progress(top_confidence)
    st.write(f"Confidence: {top_confidence:.2%}")

    st.write("Top predictions")
    for label, confidence in ranked_predictions[:5]:
        st.write(f"{label}: {confidence:.2%}")


if __name__ == "__main__":
    main()
