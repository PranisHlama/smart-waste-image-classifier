import json
import time
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MOBILENET_LABELS_PATH,
    MOBILENET_MODEL_PATH,
    TEST_DIR as CONFIG_TEST_DIR,
    TRAIN_AUGMENTED_DIR,
    VALIDATION_DIR,
)

IMG_SIZE = IMAGE_SIZE
EPOCHS = 6

TRAIN_DIR = str(TRAIN_AUGMENTED_DIR)
VAL_DIR = str(VALIDATION_DIR)
TEST_DIR = str(CONFIG_TEST_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels = "inferred",
    label_mode = "int",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels = "inferred",
    label_mode = "int",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels = "inferred",
    label_mode = "int",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed = 123
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print(f"Number of classes: {NUM_CLASSES}")
print(f"Classes: {class_names}")


# Pre process images for mobilenet
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda images, labels: (preprocess_input(images), labels))
val_ds = val_ds.map(lambda images, labels: (preprocess_input(images), labels))
test_ds = test_ds.map(lambda images, labels: (preprocess_input(images), labels))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


## MobilenetV2 Base Model

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top = False,
    weights="imagenet"
)

base_model.trainable = False

# WASTE CLASSIFIER LAYER
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile 
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train the model

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = EPOCHS
)

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

# Evaluate on test data

test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

MOBILENET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MOBILENET_MODEL_PATH)
MOBILENET_LABELS_PATH.write_text("\n".join(class_names), encoding="utf-8")
print(f"Saved model: {MOBILENET_MODEL_PATH}")
print(f"Saved labels: {MOBILENET_LABELS_PATH}")
