import pandas as pd
import numpy as np

from pathlib import Path
# from mlcroissant import Dataset

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "images"

for image_path in IMAGE_DIR.rglob("*.jpg"):
    label = image_path.parent.name
    print(label, image_path)

# ds = Dataset(jsonld=str(METADATA_PATH))

# # records = ds.records('')
# print(ds)