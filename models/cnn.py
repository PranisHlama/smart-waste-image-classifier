import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BATCH_SIZE, CNN_LABELS_PATH, CNN_MODEL_PATH, IMAGE_SIZE, MODELS_DIR, TRAIN_AUGMENTED_DIR

df = str(TRAIN_AUGMENTED_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    df,
    validation_split=0.2,
    subset="training",
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    df,
    validation_split=0.2,
    subset="validation",
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# verify the data
class_names = train_ds.class_names
print(f"Classes: {class_names}")

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.xlabel(class_names[labels[i]])
plot_path = MODELS_DIR / "img" / "cnn.png"
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")


# Create Convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

# Add dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(class_names)))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=4, validation_data=val_ds)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print(test_acc)

CNN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(CNN_MODEL_PATH)
CNN_LABELS_PATH.write_text("\n".join(class_names), encoding="utf-8")
print(f"Saved model: {CNN_MODEL_PATH}")
print(f"Saved labels: {CNN_LABELS_PATH}")
