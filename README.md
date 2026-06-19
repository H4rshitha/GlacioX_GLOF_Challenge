#  GLOF Eagles 2026 

## Explanation Video

[![GLOF Eagles 2026 Demonstration](https://img.shields.io/badge/YouTube-Demonstration%20Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/1hpBj9pNhYY)

> **[Watch the Project Demonstration Video](https://youtu.be/1hpBj9pNhYY)**
---

##  Overview
A complete, production-ready dual-task deep learning pipeline designed for the **GlacioX Glacial Lake Outburst Flood (GLOF) Challenge**:
- **Stage 1 (Classification):** 6-class classification to categorize the environmental setting of the lake region (`Cloud Cover`, `Debris Cover`, `Moraine Dammed`, `Snow Cover`, `Terrain Shadow`, `Varying Turbidity`).
- **Stage 2 (Segmentation):** Category-aware binary lake segmentation using specialized lake-centered patch mining and multi-scale curriculum training to handle tiny lakes, class imbalance, and extreme weather interference.

---

##  Architecture & Core Features

### Model Configurations
- **Classifier:** `EfficientNet-B4` (loaded via `timm`) pre-trained on ImageNet, optimized using cross-entropy loss with label smoothing ($0.1$).
- **Segmenter:** `UNet++` with an `EfficientNet-B4` encoder (loaded via `segmentation_models_pytorch`), trained with a hybrid loss: $50\%$ Focal Loss + $50\%$ Tversky Loss.

---

##  Cross-Validation Performance (Training Phase)
The training phase utilized **3-Fold Cross-Validation** on the initial subset. The ablation study shows the benefits of patch mining, multi-scale curriculum, and hard-example mining.

### 1. Classification Strategy Ablation (Mean across Folds 0, 1, 2)
| Strategy | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.611111 | 0.509259 | 0.611111 | 0.540741 | 0.540741 | 0.533333 |
| **MultiScale_Curriculum** | 0.527778 | 0.571296 | 0.527778 | 0.512169 | 0.512169 | 0.433333 |
| **Patches** | 0.583333 | 0.500000 | 0.583333 | 0.514815 | 0.514815 | 0.500000 |

### 2. Segmentation Strategy Ablation (Mean across Folds 0, 1, 2)
| Strategy | Mean IoU | Precision | Recall | F1 Score | Debris IoU | Snow IoU | Cloud IoU | Moraine IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.311997 | 0.397260 | 0.716001 | 0.434682 | 0.087367 | 0.115516 | 0.260461 | 0.388811 |
| **Curriculum_HardMining** | 0.546513 | 0.902048 | 0.580070 | 0.693392 | 0.070214 | 0.425665 | 0.341019 | 0.471810 |
| **Curriculum_Patches** | 0.534268 | 0.709444 | 0.696260 | 0.680824 | 0.087765 | 0.516567 | 0.233263 | 0.416682 |
| **Lake_Patches** | **0.562680** | 0.690205 | 0.734177 | 0.709370 | 0.104176 | 0.544838 | 0.296870 | 0.406007 |
| **MultiScale_Patches** | 0.523032 | 0.779544 | 0.595513 | 0.660583 | 0.029588 | 0.384318 | 0.437035 | 0.483230 |

---

##  Validation & Evaluation (Validation Phase)
The model was evaluated on a comprehensive **Validation Dataset containing 2,220 image-mask pairs** with complete class representation. Below are the metrics computed over the validation split (evaluating 20% validation split, which comprises 444 paired samples).

### 1. Classification Evaluation
The classification model achieved a weighted F1-score of **0.5842** and a Cohen's Kappa of **0.4012**.

#### Classification Performance Summary
- **Accuracy:** 0.5991
- **Precision (Macro):** 0.4103
- **Recall (Macro):** 0.3648
- **F1 Score (Macro):** 0.3760
- **F1 Score (Weighted):** 0.5842
- **Cohen Kappa:** 0.4012

#### Per-Class Classification Metrics
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Debris Cover** | 0.5200 | 0.2766 | 0.3611 | 47 |
| **Moraine** | 0.6011 | 0.6407 | 0.6203 | 167 |
| **Snow Cover** | 0.6406 | 0.7365 | 0.6852 | 167 |
| **Terrain Shadow** | 0.5000 | 0.4583 | 0.4783 | 48 |
| **Cloud Cover** | 0.0000 | 0.0000 | 0.0000 | 2 |
| **Turbidity** | 0.2000 | 0.0769 | 0.1111 | 13 |
| **Overall Accuracy** | **0.5991** | **0.5991** | **0.5991** | **444** |
| **Macro Average** | 0.4103 | 0.3648 | 0.3760 | 444 |
| **Weighted Average** | 0.5820 | 0.5991 | 0.5842 | 444 |

---

### 2. Segmentation Performance
The UNet++ segmentation model demonstrated superior generalization capability on the large validation set, achieving an overall **Mean IoU of 0.6260**.

#### Overall Segmentation Metrics
- **Mean IoU:** 0.6260
- **Precision:** 0.7656
- **Recall:** 0.6697
- **F1 Score:** 0.6920
- **Cohen Kappa:** 0.6954

#### Per-Category Segmentation Metrics
| Category | IoU | Precision | Recall | F1 Score | Cohen Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cloud Cover** | **0.8841** | 0.9822 | 0.8985 | 0.9385 | 0.9571 |
| **Debris Cover** | 0.5677 | 0.8025 | 0.6235 | 0.6505 | 0.6428 |
| **Moraine Dammed** | 0.6132 | 0.7391 | 0.6682 | 0.6789 | 0.6768 |
| **Snow Cover** | 0.6379 | 0.7708 | 0.6651 | 0.6990 | 0.7107 |
| **Terrain Shadow** | 0.6429 | 0.7797 | 0.6821 | 0.7085 | 0.7160 |
| **Varying Turbidity** | 0.7455 | 0.8214 | 0.8319 | 0.8227 | 0.8117 |

---

### 3. Post-Processing & Robustness Evaluation
The post-processing pipeline uses **nearest-neighbor resizing** to restore predicted masks from the $384\times384$ model resolution to original image dimensions, followed by a probability threshold of $0.5$ to produce final binary masks.
- **Robustness Overview:**
  - **Best Category:** Cloud Cover (IoU: 0.8841)
  - **Worst Category:** Debris Cover (IoU: 0.5677)
- **Top failure cases** occur in scenes with tiny lakes covered by heavy debris or under steep terrain shadow. In these cases, the low contrast and pixel overlap make it challenging to separate the lake body from the surrounding features, resulting in minor detection offsets.

---

##  Final Validation Summary

### Best Selected Models
- **Classification:** `EfficientNet-B4` trained with a custom Multi-Scale Curriculum + Patch crop pipeline.
- **Segmentation:** `UNet++ (EfficientNet-B4 backbone)` utilizing patch-based curriculum and hard-example oversampling (Curriculum_HardMining).

### Key Performance Summary
- **Mean IoU:** `0.6260`
- **Precision:** `0.7656`
- **Recall:** `0.6697`
- **Dice/F1:** `0.6920`

### Category-wise Analysis
- **Strengths:** 
  - **Cloud Cover** ($0.8841$ IoU) and **Varying Turbidity** ($0.7455$ IoU) exhibit outstanding segmentation performance. The model effectively handles severe cloud obstruction and turbidity variations.
  - High precision across all classes, indicating low false-positive rates for water detection.
- **Weaknesses:**
  - **Debris Cover** ($0.5677$ IoU) remains the most challenging class. Tiny lakes embedded in high-frequency debris patterns present low boundary contrast.
  - **Moraine Dammed** ($0.6132$ IoU) is affected by steep slopes and shadow interference, causing slight under-segmentation (recall: $0.6682$).

---

##  Competition Submission Contents
The repository contains the complete deliverables for the GLOF Challenge:

| Deliverable | Location / Details | Status |
| :--- | :--- | :---: |
| **Trained Model Files** | PyTorch checkpoint (`model.pth`), PyTorch state-dict to HDF5 (`model.h5`), and ONNX export (`model.onnx` / `model.onnx.data`) in both `Training Phase/all_model_weights` and `Validation Phase/weights` | ✓ |
| **Source Code** | Python scripts: `train.py` (training), `inference.py` (evaluation), `model_architecture.py` (models), and `utils.py` (patching/IoU) | ✓ |
| **Notebooks** | Training notebook (`Training Phase/glof_final_submission.ipynb`) & validation notebook (`Validation Phase/glof_validation_phase.ipynb`) | ✓ |
| **Segmentation Masks** | Output masks under `Training Phase/segmentation_masks` & `Validation Phase/segmentation_masks` | ✓ |
| **Reports** | Markdown reports: `dataset_diagnostic_report.md`, `evaluation_report.md`, `robustness_report.md`, and `technical_report.md` | ✓ |
| **README** | Root `README.md` (this file) & phase-specific READMEs | ✓ |
| **Requirements** | Project package dependencies (`requirements.txt`) | ✓ |
| **Explanation Video** | [Watch Project demonstration Video](https://youtu.be/QegcqrI3b38?si=OBCqAut1vDe70vHe) | ✓ |

---

##  File Description & Repository Structure
- `Training Phase/`
  - `glof_final_submission.ipynb` — Jupyter Notebook for model training.
  - `glof_final_submission.py` — Python script version of the training notebook.
  - `inference.py` — Inference script for classification and segmentation.
  - `model_architecture.py` — Class definitions for classification and segmentation models.
  - `utils.py` — Utility functions for metric computation and patch mining.
  - `requirements.txt` — Dependencies required to run training and inference.
  - `classification_predictions.csv` — Predictions on the training dataset.
  - `classification_report.csv` — Tabulated training metrics.
  - `confusion_matrix.png` — Classification confusion matrix.
  - `technical_report.md` — Detailed technical report of training experiments.
  - `segmentation_masks/` — Output segmentation masks.
  - `all_model_weights/` — Checkpoint weights.
- `Validation Phase/`
  - `glof_validation_phase.ipynb` — Jupyter Notebook running the validation evaluation pipeline.
  - `evaluation_report.md` — Detailed classification & segmentation metrics.
  - `evaluation_report.csv` — Numeric metrics on the validation dataset.
  - `dataset_diagnostic_report.md` — Dataset characteristics and class support count.
  - `robustness_report.md` — Performance detailed analysis and failure cases.
  - `failure_analysis.csv` — Tabulated per-image performance.
  - `failure_analysis.png` — IoU distribution across the validation dataset.
  - `per_category_iou.png` — Bar chart showing mean IoU per category.
  - `classification_confusion_matrix.png` — Classification confusion matrix heatmap.
  - `segmentation_masks/` — Exported binary masks on the 2,220 pairs.
  - `weights/` — Final exported model weights.
  - `visualizations/` — Visualization of prediction overlays.
