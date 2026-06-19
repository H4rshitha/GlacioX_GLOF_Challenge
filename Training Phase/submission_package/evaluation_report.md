# GLOF Eagles 2026 - Evaluation Report (V14 Final)

## 1. Dataset & Overview
* **Labeled Samples:** 60 image-mask pairs
* **Unlabeled Samples:** 575 images
* **Categories:** Cloud Cover, Debris Cover, Moraine, Snow Cover, Terrain Shadow, Turbidity

## 2. Architecture
* **Classification:** EfficientNet-B4
* **Segmentation:** UNet++ with EfficientNet-B4 encoder

## 3. Training Strategy
* **Methodology:** Curriculum_HardMining
* **Techniques:** Patch-based multi-scale curriculum, Confidence-weighted pseudo labeling, and hard-example mining for Debris/Snow classes.

## 4. Classification Results
| Metric | Score |
|---|---|
| Accuracy | 0.611111 |
| Precision | 0.582407 |
| Recall | 0.611111 |
| Macro F1 | 0.565079 |
| Weighted F1 | 0.565079 |
| Cohen Kappa | 0.533333 |

## 5. Segmentation Results
| Metric | Score |
|---|---|
| Mean IoU | 0.578419 |
| Precision | 0.841210 |
| Recall | 0.633610 |
| F1 Score | 0.720709 |

### Per-Category IoU
| Category | IoU |
|---|---|
| Debris Cover | 0.097046 |
| Snow Cover | 0.448602 |
| Cloud Cover | 0.381282 |
| Moraine | 0.474759 |

## 6. Failure Analysis
Tiny-object segmentation (especially Debris and Snow lakes occupying 0.1%–1% of the area) remains the primary bottleneck. Hard-example mining improved baseline performance significantly, but precision/recall trade-offs in turbid and shadowed regions still present challenges.

## 7. Challenge Deliverables
The submission package includes `model.pth`, `model.onnx`, `model.h5`, `submission.ipynb`, executable scripts, masks, and this report.
