# Alzheimer MRI Classification using DenseNet169

## 📌 Project Overview
This project implements a **deep learning–based convolutional neural network (CNN)** to classify **Alzheimer’s disease stages** from **brain MRI images**.  
The model leverages **DenseNet169 pretrained on ImageNet** using **transfer learning** to extract rich visual features and perform **multi-class classification** across different dementia stages.

The objective is to analyze MRI brain slices and predict the cognitive condition of a subject among four categories:
- Non Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

---

## 📂 Dataset
**Source:**  
Kaggle – Alzheimer MRI 4 Classes Dataset  
🔗 https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset/data

### Dataset Description
The dataset consists of **2D MRI brain slices** extracted from MRI scans and categorized into four dementia stages:

| Class | Number of Subjects |
|-----|-------------------|
| Non Demented | 100 |
| Very Mild Demented | 70 |
| Mild Demented | 28 |
| Moderate Demented | 2 |

⚠️ **Important Note:**  
- Each subject contributes multiple MRI slices.
- The dataset is **highly imbalanced at the subject level**, particularly for the *Moderate Demented* class.
- The original dataset had ordering issues; this version merges and **randomly splits images into train, validation, and test sets**, reducing slice-order bias.

---

## 🛠️ Project Workflow

### 1️⃣ Data Preparation and Splitting
- All images were initially stored class-wise.
- The dataset was **manually split** into:
  - **70% Training**
  - **15% Validation**
  - **15% Test**
- This ensured a clean separation to avoid data leakage.

```text
dataset/
├── train/
├── val/
└── test/

2️⃣ Data Preprocessing and Augmentation

Images resized to 224 × 224

Pixel normalization (rescale=1./255)

Data augmentation applied only to training data:

Rotation

Zoom

Horizontal and vertical flipping

3️⃣ Model Architecture

Base Model: DenseNet169 (ImageNet pretrained, include_top=False)

Pooling: Global Average Pooling (GAP)

Classifier Head:

Dense layers with ReLU activation

Batch Normalization

Dropout for regularization

Training Strategy:

Initially freeze all DenseNet layers

Fine-tune the last few layers for domain adaptation

4️⃣ Model Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Metrics Tracked:

Accuracy

AUC (Area Under ROC Curve)

Callbacks:

EarlyStopping (monitored on val_auc)

ModelCheckpoint (saves best model based on val_auc)

5️⃣ Model Evaluation

Evaluation was performed on a held-out test set, never seen during training or validation.

📊 Results and Performance
🔹 Training & Validation (Last Epochs Summary)

Training AUC ≈ 0.98

Validation AUC peaked at 0.9675

No significant gap between training and validation metrics → no overfitting








##🔹 Test Set Performance
Test Accuracy: 83.37%
Test AUC: 96.57%
The strong alignment between validation and test AUC indicates good generalization.

##🔍 Single Image Inference Results
| Test Case | True Class         | Prediction         | Confidence |
| --------- | ------------------ | ------------------ | ---------- |
| Case 1    | Non Demented       | Non Demented       | 83.36%     |
| Case 2    | Mild Demented      | Mild Demented      | 90.50%     |
| Case 3    | Moderate Demented  | Non Demented       | 36.40%     |
| Case 4    | Very Mild Demented | Very Mild Demented | 82.88%     |


##Interpretation:
- The model performs reliably for Non, Very Mild, and Mild Demented cases.
- The Moderate Demented class is challenging, primarily due to having MRI slices from only two subjects.
- The low confidence (36.4%) reflects model uncertainty rather than overconfident misclassification, which is desirable in medical AI.

⚠️ **Limitations**
- Severe class imbalance at subject level, especially for Moderate Demented
- Slice-based learning does not capture full 3D anatomical context
- Softmax confidence scores are not calibrated medical probabilities

##📌 Key Takeaways

- Transfer learning with DenseNet169 is effective for MRI-based dementia classification
- AUC is a more reliable metric than accuracy for imbalanced medical datasets
- Dataset quality and subject diversity are critical for clinical generalization

