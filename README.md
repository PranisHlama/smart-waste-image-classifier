# Smart Waste Image Classifier

Smart Waste Image Classifier is a Python image classification project that predicts the waste category of an uploaded image. It includes data preparation scripts, image augmentation, a custom CNN model, a MobileNetV2 transfer-learning model, and a Streamlit demo app for testing predictions.

The dataset is organized as class folders under `data/images` and currently contains 4,752 images across 9 classes:

- `1-Cardboard`
- `2-Food Organics`
- `3-Glass`
- `4-Metal`
- `5-Miscellaneous Trash`
- `6-Paper`
- `7-Plastic`
- `8-Textile Trash`
- `9-Vegetation`

## Project Structure

```text
.
├── app.py                         # Streamlit prediction app
├── config.py                      # Shared paths and training settings
├── requirements.txt               # Python dependencies
├── data/
│   ├── images/                    # Original image dataset, stored with Git LFS
│   ├── train/                     # Generated training split
│   ├── validation/                # Generated validation split
│   ├── test/                      # Generated test split
│   └── train_augmented/           # Generated augmented training images
├── models/
│   ├── cnn.py                     # Custom CNN training script
│   ├── mobilenet.py               # MobileNetV2 training script
│   ├── cnn.keras                  # Generated custom CNN model
│   ├── mobilenet.keras            # Generated MobileNetV2 model
│   ├── cnn_labels.txt             # Generated labels for the CNN model
│   └── mobilenet_labels.txt       # Generated labels for the MobileNetV2 model
└── src/
    ├── augment_images.py          # Creates augmented training images
    ├── preprocessing.py           # Image loading and augmentation helpers
    ├── split_dataset.py           # Splits images into train/validation/test
    └── train.py                   # Basic dataset loading check
```

Generated folders and model files may not exist until you run the preparation and training commands.

## Git LFS Dataset Files

The project images are tracked with Git LFS. The `.gitattributes` file marks `*.jpg` files as LFS objects, so a normal clone may only download small pointer files until LFS is installed and pulled.

Install Git LFS if needed:

```bash
git lfs install
```

After cloning the repository, download the actual image files:

```bash
git lfs pull
```

To confirm the dataset is available, check that the files under `data/images` are real JPG images rather than LFS pointer text files.

## Requirements

- Python 3.11 or compatible Python 3 version
- Git LFS
- Enough disk space for the image dataset, augmented images, and trained model files
- Optional but recommended: a virtual environment

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

If your system has both Python 2 and Python 3 installed, use `python3` instead of `python`.

## Setup From a Fresh Clone

1. Clone the repository.

```bash
git clone <repository-url>
cd smart_waste_image_classifier
```

2. Install and pull Git LFS files.

```bash
git lfs install
git lfs pull
```

3. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies.

```bash
python -m pip install -r requirements.txt
```

5. Split the original dataset into training, validation, and test folders.

```bash
python src/split_dataset.py --overwrite
```

By default this creates:

- `data/train`
- `data/validation`
- `data/test`

The default split is 70% training, 15% validation, and 15% test.

6. Create augmented training images.

```bash
python src/augment_images.py --include-originals
```

This writes augmented images to `data/train_augmented`. The training scripts use this folder.

## Train the Models

Train the MobileNetV2 transfer-learning model:

```bash
python models/mobilenet.py
```

This creates:

- `models/mobilenet.keras`
- `models/mobilenet_labels.txt`

Train the custom CNN model:

```bash
python models/cnn.py
```

This creates:

- `models/cnn.keras`
- `models/cnn_labels.txt`

MobileNetV2 usually gives better results because it starts from ImageNet pretrained weights. The custom CNN is included as a baseline model for comparison.

## Run the Streamlit App

After at least one model has been trained, start the demo app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a waste image, and choose either `MobileNetV2` or `Custom CNN` from the model selector. If a selected model has not been trained yet, the app will show the training command needed for that model.

## Common Workflow

For a full run from prepared source images:

```bash
git lfs pull
python -m pip install -r requirements.txt
python src/split_dataset.py --overwrite
python src/augment_images.py --include-originals
python models/mobilenet.py
streamlit run app.py
```

To also train the baseline CNN:

```bash
python models/cnn.py
```

## Notes

- `data/images` is the original class-folder dataset.
- `data/train`, `data/validation`, `data/test`, and `data/train_augmented` are generated from the original dataset.
- `models/*.keras` and `models/*_labels.txt` are generated by training scripts.
- The augmentation script creates horizontal flips, small rotations, brightness changes, and contrast changes.
- The project uses 224 x 224 RGB images, configured in `config.py`.
