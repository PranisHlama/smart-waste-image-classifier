from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
TRAIN_AUGMENTED_DIR = DATA_DIR / "train_augmented"
METADATA_PATH = DATA_DIR / "waste-classification-metadata.json"

MODELS_DIR = BASE_DIR / "models"
CNN_MODEL_PATH = MODELS_DIR / "cnn.keras"
MOBILENET_MODEL_PATH = MODELS_DIR / "mobilenet.keras"
CNN_LABELS_PATH = MODELS_DIR / "cnn_labels.txt"
MOBILENET_LABELS_PATH = MODELS_DIR / "mobilenet_labels.txt"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"
PLOTS_DIR = RESULTS_DIR / "plots"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
