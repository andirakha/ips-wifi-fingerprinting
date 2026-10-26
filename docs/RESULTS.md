# Results

Aggregate metrics produced by `notebooks/main.ipynb` on the full dataset
(360 fingerprints, 10 locations, 9 reference points per location, 4 devices, 12 target APs).
Model: `KNeighborsClassifier(weights='distance', metric='manhattan')` on standardized RSSI features.

## 1. Choosing K (manual split: points 1-5 train, points 6-9 validation)

| K | Accuracy |
|---|---|
| 1 | 0.9000 |
| 3 | 0.9000 |
| 5 | 0.9062 |
| 7 | 0.9062 |
| 9 | 0.9187 |
| **11** | **0.9250** |
| 13 | 0.9125 |
| 15 | 0.9125 |

K=11 was selected. Macro-averaged precision / recall / F1 at K=11 are 0.93 / 0.93 / 0.92 (160 validation samples).

## 2. Group K-Fold (5 folds, grouped by reference point)

| Fold | Accuracy |
|---|---|
| 1 | 0.8125 |
| 2 | 0.8875 |
| 3 | 0.9000 |
| 4 | 0.8750 |
| 5 | 0.9000 |
| **Mean** | **0.8750** |

## 3. Leave-One-Group-Out (one reference point held out per iteration)

Per-point accuracies: 0.8500, 0.9000, 0.8500, 0.7500, 0.9000, 0.9500, 0.9000, 0.9500, 0.9250.
**Mean: 0.8861.**

## Reading the numbers

All three evaluations hold out whole reference points, so none of them tests on positions seen in training.
K was chosen on the same manual validation split that reports 0.9250, so that figure is slightly optimistic;
the Group K-Fold and LOGO means (about 0.88) are the more reliable estimates of accuracy at unseen positions.

Per-location reports and confusion matrices are produced when you run the notebook locally; they name real
rooms, so they are not committed.
