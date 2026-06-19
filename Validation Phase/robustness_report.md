# Robustness & Failure Analysis Report

- **Best Performing Category**: Cloud Cover (IoU: 0.8841)
- **Worst Performing Category**: Debris Cover (IoU: 0.5677)

## Category Robustness Details
### Cloud Cover
- Mean IoU: 0.8841
- Precision: 0.9822
- Recall: 0.8985
### Debris Cover
- Mean IoU: 0.5677
- Precision: 0.8025
- Recall: 0.6235
### Moraine Dammed
- Mean IoU: 0.6132
- Precision: 0.7391
- Recall: 0.6682
### Snow Cover
- Mean IoU: 0.6379
- Precision: 0.7708
- Recall: 0.6651
### Terrain Shadow
- Mean IoU: 0.6429
- Precision: 0.7797
- Recall: 0.6821
### Varying Turbidity
- Mean IoU: 0.7455
- Precision: 0.8214
- Recall: 0.8319

## Top 20 Worst Predictions (Failure Cases)
| Stem | True Category | Predicted Class | IoU |
|---|---|---|---|
| 1182 | Snow Cover | Snow Cover | 0.0000 |
| 1877 | Terrain Shadow | Terraine Shadow | 0.0000 |
| 1504 | Varying Turbidity | Moraine | 0.0000 |
| 736 | Debris Cover | Terraine Shadow | 0.0000 |
| 1465 | Moraine Dammed | Moraine | 0.0000 |
| 942 | Moraine Dammed | Moraine | 0.0000 |
| 277 | Moraine Dammed | Snow Cover | 0.0000 |
| 1421 | Snow Cover | Moraine | 0.0000 |
| 91 | Snow Cover | Snow Cover | 0.0000 |
| 1876 | Moraine Dammed | Snow Cover | 0.0000 |
| 1784 | Debris Cover | Terraine Shadow | 0.0000 |
| 2367 | Debris Cover | Debris Cover | 0.0000 |
| 1057 | Moraine Dammed | Snow Cover | 0.0000 |
| 160 | Moraine Dammed | Moraine | 0.0000 |
| 46 | Snow Cover | Snow Cover | 0.0000 |
| 214 | Moraine Dammed | Moraine | 0.0000 |
| 372 | Snow Cover | Snow Cover | 0.0000 |
| 2322 | Debris Cover | Snow Cover | 0.0000 |
| 133 | Snow Cover | Moraine | 0.0000 |
| 980 | Moraine Dammed | Moraine | 0.0000 |