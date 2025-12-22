# Skin Lesion Classification – Deep Learning Based Disease Detection with Grad-CAM

This repository documents the development of a **deep learning–based skin lesion classification system** that automatically detects and classifies different types of skin lesions from dermoscopic images.  
The project focuses on end-to-end model development including data preprocessing, class imbalance handling, CNN-based training, evaluation, inference, and **visual explainability using Grad-CAM**.

---

## 📅 Project Overview

### Phase 1: Dataset Preparation & Preprocessing
The initial phase focuses on preparing dermoscopic image data and metadata for robust model training.

**Key tasks performed:**
- Loading and validating image datasets (HAM10000)
- Cleaning and preprocessing metadata (CSV files)
- Image resizing, normalization, and format standardization
- Handling missing labels and invalid image entries
- Train–validation split using `ImageDataGenerator`

---

### Phase 2: Model Development & Training
This phase involves building and training convolutional neural network models for multi-class skin lesion classification.

**Core highlights:**
- Transfer learning using pre-trained CNN architectures  
  - EfficientNetB0  
  - MobileNetV2 (CPU-optimized)
- Custom classification head for multi-class prediction
- Handling severe class imbalance using computed class weights
- Batch-by-batch CPU-optimized training
- Early stopping and learning rate scheduling

**Skin lesion classes:**
- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

---

### Phase 3: Model Evaluation & Inference
In the final phase, trained models are evaluated and used for prediction on unseen images.

**Key features:**
- Validation accuracy and loss analysis
- Per-image prediction with confidence scores
- Class-wise probability output for interpretability
- Single-image inference using a prediction script
- Model persistence using the `.keras` format

---

## Grad-CAM – Model Explainability & Visual Interpretation

To improve transparency and interpretability, the project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM visually highlights the **regions of the lesion image that most influenced the model’s prediction**, helping explain *why* a specific class was chosen.

### Why Grad-CAM?
- Deep learning models often behave as black boxes
- Explainability is especially important in medical imaging
- Helps verify that the model focuses on lesion regions rather than background
- Aids in analyzing correct and incorrect predictions

### Grad-CAM Capabilities
- Compatible with EfficientNet and MobileNet models
- Works with single-image inference
- Generates heatmaps highlighting important regions
- Overlays heatmaps on original images
- Provides visual confidence in model behavior

### 🔹 Grad-CAM Output
For each input image:
- Original image
- Grad-CAM heatmap
- Overlay visualization (image + heatmap)
- Predicted class with confidence score

> **Note:** Grad-CAM is used for interpretability and research purposes only and does not replace professional medical diagnosis.

---

## 🛠 Tools & Technologies
- Python
- TensorFlow / Keras
- OpenCV
- pandas, NumPy
- scikit-learn
- Jupyter Notebook / VS Code

---

## Output to User
For each uploaded skin lesion image, the system provides:
- Predicted lesion class
- Probability score for each disease category
- Final predicted diagnosis with confidence percentage
- Optional Grad-CAM visualization for explainability

This output enables better understanding of predictions and supports educational and research use cases.

---

## Use Cases
- Academic and research projects in medical imaging
- Learning and experimentation with CNNs and transfer learning
- Portfolio-level ML / AI demonstration
- Baseline system for further research in skin disease detection

---

## Future Scope
- Full Grad-CAM integration into web interface
- Multi-model ensemble for improved accuracy
- Streamlit or Flask-based deployment
- Higher-resolution fine-tuning
- Automated report generation with predictions and heatmaps

---

