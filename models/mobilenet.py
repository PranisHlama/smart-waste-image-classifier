import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6

TRAIN_DIR = '../data/train_augmented'
VAL_DIR = '../data/validation'
TEST_DIR = '../data/test'

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
    loss="categorical_crossentropy",
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

# Precision Recall F1-Score and confusion matrix
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

model.save("mobilenet_waste_classifier.keras")

metadata = {
    "class_names": class_names,
    "image_size": IMG_SIZE,
    "preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input"
}


with open("mobilenet_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Saved model to mobilenet_waste_classifier.keras")
print("Saved metadata to mobilenet_metadata.json")