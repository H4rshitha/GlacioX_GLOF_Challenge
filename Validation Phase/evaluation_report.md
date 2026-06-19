# TEST_V17 Evaluation Report

## Classification Metrics
- **Accuracy**: 0.5991
- **Precision (Macro)**: 0.4103
- **Recall (Macro)**: 0.3648
- **F1 Score (Macro)**: 0.3760
- **F1 Score (Weighted)**: 0.5842
- **Cohen Kappa**: 0.4012

### Classification Report
|                 |   precision |    recall |   f1-score |    support |
|:----------------|------------:|----------:|-----------:|-----------:|
| Cloud Cover     |    0        | 0         |   0        |   2        |
| Debris Cover    |    0.52     | 0.276596  |   0.361111 |  47        |
| Moraine         |    0.601124 | 0.640719  |   0.62029  | 167        |
| Snow Cover      |    0.640625 | 0.736527  |   0.685237 | 167        |
| Terraine Shadow |    0.5      | 0.458333  |   0.478261 |  48        |
| Turbidity       |    0.2      | 0.0769231 |   0.111111 |  13        |
| accuracy        |    0.599099 | 0.599099  |   0.599099 |   0.599099 |
| macro avg       |    0.410291 | 0.36485   |   0.376002 | 444        |
| weighted avg    |    0.582009 | 0.599099  |   0.584226 | 444        |

## Segmentation Metrics (Overall)
- **IoU**: 0.6260
- **Precision**: 0.7656
- **Recall**: 0.6697
- **F1 Score**: 0.6920
- **Cohen Kappa**: 0.6954

## Per-Category Segmentation Performance
| Category | IoU | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|---|
| Cloud Cover | 0.8841 | 0.9822 | 0.8985 | 0.9385 | 0.9571 |
| Debris Cover | 0.5677 | 0.8025 | 0.6235 | 0.6505 | 0.6428 |
| Moraine Dammed | 0.6132 | 0.7391 | 0.6682 | 0.6789 | 0.6768 |
| Snow Cover | 0.6379 | 0.7708 | 0.6651 | 0.6990 | 0.7107 |
| Terrain Shadow | 0.6429 | 0.7797 | 0.6821 | 0.7085 | 0.7160 |
| Varying Turbidity | 0.7455 | 0.8214 | 0.8319 | 0.8227 | 0.8117 |

## Robustness Overview
- **Best Category**: Cloud Cover (IoU: 0.8841)
- **Worst Category**: Debris Cover (IoU: 0.5677)