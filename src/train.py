from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_DIR, VALIDATION_DIR
from preprocessing import load_dataset_as_arrays

X_train, y_train = load_dataset_as_arrays(TRAIN_DIR)
X_validation, y_validation = load_dataset_as_arrays(VALIDATION_DIR)

print(X_train.shape)
print(y_train.shape)
