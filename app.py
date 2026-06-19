from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def list_available_models() -> dict[str, tuple[Path, Path]]:
    model_pairs: dict[str, tuple[Path, Path]] = {}
    for model_path in sorted(MODELS_DIR.glob("*.keras")):
        candidate_metadata = [
            MODELS_DIR / f"{model_path.stem}_metadata.json",
            MODELS_DIR / f"{model_path.stem.replace('_waste_classifier', '')}_metadata.json",
        ]
        for metadata_path in candidate_metadata:
            if metadata_path.exists():
                model_pairs[model_path.stem] = (model_path, metadata_path)
                break
    return model_pairs


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_metadata(metadata_path: str) -> dict:
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_image(image: Image.Image, image_size: tuple[int, int], preprocessing: str) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)

    if "mobilenet_v2.preprocess_input" in preprocessing:
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    else:
        array = array / 255.0

    return np.expand_dims(array, axis=0)


def pretty_label(label: str) -> str:
    return label.split("-", 1)[-1]


def main() -> None:
    st.set_page_config(page_title="Smart Waste Classifier", layout="centered")
    st.title("Smart Waste Classifier")
    st.caption("Upload a waste image and run inference with a trained model.")

    available_models = list_available_models()
    if not available_models:
        st.error("No trained model + metadata pair found in the models directory.")
        st.stop()

    default_model = (
        "mobilenet_waste_classifier"
        if "mobilenet_waste_classifier" in available_models
        else next(iter(available_models))
    )

    selected_model = st.selectbox(
        "Model",
        options=list(available_models.keys()),
        index=list(available_models.keys()).index(default_model),
        format_func=lambda name: name.replace("_", " ").title(),
    )

    model_path, metadata_path = available_models[selected_model]
    metadata = load_metadata(str(metadata_path))
    model = load_model(str(model_path))

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    try:
        image = Image.open(uploaded_file)
    except (OSError, ValueError):
        st.error("Unable to read the uploaded image.")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)

    if not st.button("Predict", type="primary"):
        return

    image_size = tuple(metadata["image_size"])
    input_batch = prepare_image(image, image_size, metadata["preprocessing"])
    probabilities = model.predict(input_batch, verbose=0)[0]

    class_names = metadata["class_names"]
    top_indices = np.argsort(probabilities)[::-1][:3]
    predicted_index = int(top_indices[0])

    st.success(f"Prediction: {pretty_label(class_names[predicted_index])}")
    st.metric("Confidence", f"{probabilities[predicted_index] * 100:.2f}%")

    rows = [
        {
            "Class": pretty_label(class_names[index]),
            "Probability": float(probabilities[index]),
        }
        for index in top_indices
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    chart_data = {
        pretty_label(label): float(score)
        for label, score in zip(class_names, probabilities)
    }
    st.bar_chart(chart_data)


if __name__ == "__main__":
    main()
