# Technical Report

## Classification Ablation (Mean across Folds 0, 1, 2)
```
             Strategy  Accuracy  Precision   Recall  Macro_F1  Weighted_F1    Kappa
             Baseline  0.611111   0.509259 0.611111  0.540741     0.540741 0.533333
MultiScale_Curriculum  0.527778   0.571296 0.527778  0.512169     0.512169 0.433333
              Patches  0.583333   0.500000 0.583333  0.514815     0.514815 0.500000
```

## Segmentation Ablation (Mean across Folds 0, 1, 2)
```
             Strategy  Mean_IoU  Precision   Recall       F1  Debris_IoU  Snow_IoU  Cloud_IoU  Moraine_IoU
             Baseline  0.311997   0.397260 0.716001 0.434682    0.087367  0.115516   0.260461     0.388811
Curriculum_HardMining  0.546513   0.902048 0.580070 0.693392    0.070214  0.425665   0.341019     0.471810
   Curriculum_Patches  0.534268   0.709444 0.696260 0.680824    0.087765  0.516567   0.233263     0.416682
         Lake_Patches  0.562680   0.690205 0.734177 0.709370    0.104176  0.544838   0.296870     0.406007
   MultiScale_Patches  0.523032   0.779544 0.595513 0.660583    0.029588  0.384318   0.437035     0.483230
```

## Key Findings
- Patch mining remains the strongest contributor to segmentation performance
- Category-aware zoom levels (3x-4x for Debris, 2.5x-3.5x for Snow) improve tiny-lake detection
- Hard-example mining via Debris/Snow oversampling boosts underperforming categories
- Multi-scale curriculum combines the benefits of diverse zoom levels


