# Technical Report

# GLOF Challenge Validation Phase Technical Report

This technical report documents the dataset, pipeline architecture, training details, optimization strategy, evaluation results, and failure mode analysis for the validation phase of the **Glacial Lake Outburst Flood (GLOF)** detection challenge.

---

## 1. Dataset Characteristics

The validation phase dataset comprises **2220 matched image-mask pairs** with zero mismatches. The instances are distributed across 6 categories representing different weather, terrain, and turbidity conditions.

| Category (Test Name) | Category (Train Name) | Images | Masks | Percentage |
| :--- | :--- | :---: | :---: | :---: |
| Cloud Cover | Cloud Cover | 12 | 12 | 0.54% |
| Debris Cover | Debris Cover | 234 | 234 | 10.54% |
| Moraine Dammed | Moraine | 835 | 835 | 37.61% |
| Snow Cover | Snow Cover | 834 | 834 | 37.57% |
| Terrain Shadow | Terraine Shadow | 239 | 239 | 10.77% |
| Varying Turbidity | Turbidity | 66 | 66 | 2.97% |
| **TOTAL** | | **2220** | **2220** | **100%** |

*Note: The dataset exhibits severe class imbalance, with Moraine Dammed and Snow Cover representing ~75% of the total dataset, whereas Cloud Cover and Varying Turbidity are highly underrepresented.*

---

## 2. Pipeline & Model Architecture

The GLOF detection pipeline is structured as a dual-model, multi-task learning workflow consisting of a classifier and a segmenter.

```mermaid
graph TD
    A[Input Image] --> B[Preprocessing & Resize 384x384]
    B --> C[EfficientNet-B4 Classifier]
    B --> D[UNet++ Segmenter]
    C --> E[Class Prediction 1 of 6]
    D --> F[Binary Segmentation Mask]
