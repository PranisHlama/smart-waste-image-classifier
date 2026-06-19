from pathlib import Path
from preprocessing import load_dataset_as_arrays

X_train, y_train = load_dataset_as_arrays(Path("../data/train"))
X_validation, y_validation = load_dataset_as_arrays(Path("../data/validation"))

print(X_train.shape)
print(y_train.shape)
