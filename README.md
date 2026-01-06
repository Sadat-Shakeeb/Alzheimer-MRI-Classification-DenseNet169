# Alzheimer MRI Classification using DenseNet169

A deep learning project to classify Alzheimer’s disease stages from 2D brain MRI slices using transfer learning with DenseNet169 (pretrained on ImageNet). The model performs multi-class classification across four dementia stages.

- Classes: Non Demented, Very Mild Demented, Mild Demented, Moderate Demented
- Input size: 224 × 224 RGB images (rescaled)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
  - [Source](#source)
  - [Description](#description)
  - [Directory Structure & Split](#directory-structure--split)
- [Preprocessing & Augmentation](#preprocessing--augmentation)
- [Model](#model)
- [Training](#training)
- [Evaluation & Results](#evaluation--results)
  - [Test set summary](#test-set-summary)
  - [Single image inference examples](#single-image-inference-examples)
- [Limitations](#limitations)
- [Usage / Reproducibility](#usage--reproducibility)
- [Tips to Improve](#tips-to-improve)
- [License & Contact](#license--contact)

---

## Project Overview

This repository demonstrates using DenseNet169 with transfer learning to extract visual features from MRI slices and classify dementia stage. The approach uses a pretrained DenseNet169 base (include_top=False), a Global Average Pooling layer, and a small classifier head. The training workflow first freezes the base model and later fine-tunes the top DenseNet blocks.

## Dataset

### Source
Kaggle — Alzheimer MRI 4 Classes Dataset  
https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset/data

### Description

- The dataset contains 2D MRI brain slices from multiple subjects, labeled into four categories.
- Subject-level counts (approximate, taken from dataset metadata used here):

| Class                | Number of Subjects |
|---------------------:|-------------------:|
| Non Demented         | 100                |
| Very Mild Demented   | 70                 |
| Mild Demented        | 28                 |
| Moderate Demented    | 2                  |

Important notes:
- Each subject contributes multiple slices. This means slice-level counts are larger and class imbalance at the subject level can affect generalization.
- The dataset used here was shuffled and split into train/val/test to reduce slice-order bias.

### Directory structure & split

```text
dataset/
├── train/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   └── ModerateDemented/
├── val/
└── test/
```

Split used:
- 70% training
- 15% validation
- 15% test

---

## Preprocessing & Augmentation

- Resize images to 224 × 224
- Pixel scaling: rescale=1./255
- Data augmentation (applied to training set only):
  - Random rotations
  - Random zoom
  - Horizontal and vertical flips (as appropriate)
  - Any other augmentations should be applied carefully for medical images

Example preprocessing pipeline (Keras-style):

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)
```

---

## Model

- Base model: DenseNet169 (weights='imagenet', include_top=False)
- Pooling: GlobalAveragePooling2D
- Head:
  - Dense layers with ReLU
  - BatchNormalization
  - Dropout for regularization
  - Final Dense with softmax (4 units)
- Training strategy:
  1. Freeze all DenseNet169 layers, train classifier head
  2. Unfreeze last few DenseNet blocks and fine-tune with a reduced learning rate

---

## Training

- Optimizer: Adam (reduce LR when fine-tuning)
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, AUC (per-class / macro AUC can be useful)
- Callbacks used:
  - EarlyStopping (monitoring validation AUC with patience)
  - ModelCheckpoint (save best model by validation AUC)
  - ReduceLROnPlateau (optional)

Suggested hyperparameters:
- Batch size: 16–64 (depending on GPU memory)
- Initial epochs: 10–30 for head, then 10–20 for fine-tuning
- Initial learning rate: 1e-4 (head), 1e-5 (fine-tuning)

---

## Evaluation & Results

Evaluation performed on a held-out test set (never used during training/validation).

Training & validation summary (final/approximate):
- Training AUC ≈ 0.98
- Validation AUC peak ≈ 0.9675

Test set performance:
- Test Accuracy: 83.37%
- Test AUC: 96.57%

The small gap between validation and test AUC suggests reasonable generalization on the test slices, however subject-level imbalance must be considered.

### Single image inference examples

| Test Case | True Class         | Prediction         | Confidence |
|-----------|--------------------|--------------------|-----------:|
| Case 1    | Non Demented       | Non Demented       | 83.36%     |
| Case 2    | Mild Demented      | Mild Demented      | 90.50%     |
| Case 3    | Moderate Demented  | Non Demented       | 36.40%     |
| Case 4    | Very Mild Demented | Very Mild Demented | 82.88%     |

Interpretation:
- Model performs well for Non, Very Mild, and Mild Demented classes.
- Moderate Demented is underrepresented (only 2 subjects), leading to poor performance and low confidence on those examples.

---

## Limitations

- Severe class imbalance at subject level, especially Moderate Demented (only a couple of subjects).
- Slice-based (2D) approach loses 3D anatomical context present in full scans.
- Softmax confidence is not a calibrated medical probability.
- Potential data leakage risk if splits are done slice-wise without considering subject-level separation — ensure subject-wise split.
- Results are dataset-dependent; real-world clinical performance requires more diverse, multi-center data and robust validation.

---

## Usage / Reproducibility

Requirements:
- Python 3.8+
- TensorFlow 2.x (or compatible)
- Other packages: numpy, pandas, scikit-learn, matplotlib, pillow, etc.
A `requirements.txt` is recommended to fix exact versions.

Quick start (example commands — adjust scripts and flags to match your repo):

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py --data_dir dataset/ --epochs 50 --batch_size 32 --save_dir checkpoints/

# Evaluate
python evaluate.py --model checkpoints/best_model.h5 --data_dir dataset/test/

# Inference (single image)
python infer.py --image path/to/image.jpg --model checkpoints/best_model.h5
```

Reproducibility tips:
- Fix random seeds for numpy, TensorFlow, and python random
- Save the full training config (optimizer, LR schedule, augmentation parameters)
- Use subject-wise splits and document how the split was created

---

## Tips to Improve

- Increase subject-level diversity in the Moderate Demented class (collect more subjects).
- Consider 3D CNNs or slice aggregation methods (e.g., per-subject voting, RNN over slice sequence) to capture volumetric context.
- Use class-weighting, focal loss, or oversampling at subject-level (not just slice-level) to reduce bias.
- Calibrate model probabilities (e.g., temperature scaling) before using them as medical decision support.
- Cross-validation with subject-wise folds for more robust performance estimates.

---

## License & Contact

This project is provided as-is for research/demo purposes. It is not a clinical product. Use responsibly.

Author: Sadat-Shakeeb  
Repository: https://github.com/Sadat-Shakeeb/Alzheimer-MRI-Classification-DenseNet169
