# CS-4375 Assignment 2 — Neural Networks

## Overview

This project trains and evaluates multiple neural network models on the **Wine Quality** dataset (Red & White) from the UCI Machine Learning Repository. All combinations of the hyperparameters below are explored, producing performance metrics and loss-curve plots.

---

## Dataset

**Wine Quality (Red & White)**  
UCI ML Repository ID: `186`  
URL: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

Fetched programmatically using the `ucimlrepo` Python package. The combined dataset contains **5,320 samples** (after deduplication) with **12 physicochemical input features** and one output feature (`quality`, integer score 3–9).

---

## Requirements

- Python 3.8+

Install all dependencies via:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy
pandas
matplotlib
scikit-learn
ucimlrepo
```

---

## How to Run

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the program
python NeuralNet.py
```

---

## Output

Running the program produces:

| Output | Description |
|---|---|
| Console table | Hyperparameters, train/test accuracy, train/test MSE for all 48 models |
| `model_history_all.png` | Loss vs. epoch for all 48 models on a single figure |
| `model_history_by_activation.png` | Loss vs. epoch split into 3 subplots by activation function |

---

## Hyperparameter Grid

| Parameter | Values Tested |
|---|---|
| Activation Function | `logistic` (sigmoid), `tanh`, `relu` |
| Learning Rate | `0.01`, `0.1` |
| Max Iterations (Epochs) | `100`, `200` |
| Hidden Layers | `2`, `3` |
| L2 Regularization (alpha) | `0.0001`, `0.01` |
| Neurons per Hidden Layer | `12` (fixed) |

**Total combinations: 3 × 2 × 2 × 2 × 2 = 48 models**

---

## Pre-processing Steps

1. **Duplicate removal** — Exact duplicate rows dropped (5,320 unique rows remain).
2. **Feature/target split** — `quality` used as the target; all other columns used as features.
3. **Missing value handling** — Numeric NaN values filled with the column mean.
4. **Categorical encoding** — Non-numeric columns label-encoded with `LabelEncoder`.
5. **Standardization** — Z-score normalization applied to all features.

---

## Results Summary

| Metric | Model | Value |
|---|---|---|
| **Best test accuracy** | `logistic_lr0.01_ep200_hl2_alpha0.01` | **58.08%** |
| **Worst test accuracy** | `tanh_lr0.1_ep100_hl2_alpha0.01` | **51.13%** |

### Average Test Accuracy by Activation Function

| Activation | Avg. Test Accuracy |
|---|---|
| `logistic` | 0.5592 |
| `relu` | 0.5471 |
| `tanh` | 0.5393 |

---

## Analysis

### Activation Functions
**Logistic (sigmoid)** achieved the highest average test accuracy (55.92%) and produced the single best model. **ReLU** ranked second (54.71%), while **tanh** ranked last (53.93%). The ~2 percentage point gap between logistic and tanh is meaningful given the narrow overall accuracy range.

### Learning Rate
A learning rate of **0.01** consistently outperformed **0.1** — particularly for logistic and tanh activations. The best overall model used lr=0.01. Higher learning rates (0.1) increased the risk of overshooting minima, as seen with tanh where the worst model (tanh_lr0.1_ep100_hl2_alpha0.01, 51.13%) was also a high-lr configuration. ReLU was more resilient to this effect.

### Number of Epochs (100 vs. 200)
With early stopping removed, the epoch budget now has a genuine effect. Several models improved meaningfully at 200 epochs — for example, logistic_lr0.01_ep200_hl2_alpha0.01 (58.08%) outperforms its 100-epoch counterpart (57.33%). This confirms 100 epochs was sometimes insufficient for convergence at lr=0.01, and the extra budget was put to use.

### Depth (2 vs. 3 Hidden Layers)
Results were mixed. For logistic at lr=0.01, 2 hidden layers slightly outperformed 3. For tanh and relu, 3 layers sometimes produced higher training accuracy but did not reliably improve test accuracy, suggesting some overfitting with deeper networks with this amount of neurons.

### L2 Regularization (alpha)
The effect of alpha was small but visible. In the logistic group, alpha=0.01 produced the best single model (58.08%), showing that a regularization penalty helped generalization on this noisy dataset. In the tanh group, higher alpha sometimes hurt — notably the worst model overall used alpha=0.01 with lr=0.1. Adding alpha though helped with overfitting massivly though. Before adding alpha in our model, overfitting was a problem. Some models had 91% train accuracy and 50% test accuracy. 

### Overfitting
Several tanh and relu models show a noticeable train/test accuracy gap (e.g., tanh_lr0.01_ep200_hl3_alpha0.0001: train 62.92%, test 54.42%), indicating overfitting. Logistic models showed more balanced train/test gaps, contributing to stronger generalization.

### Overall
All models fall in a ~51-58% accuracy range, this shows the true difficulty of the 7-class problem. The class distribution is heavily skewed toward quality scores 5 and 6, making minority classes (3, 4, 8, 9) hard to predict. Logistic activation with a low learning rate, 2 hidden layers, and moderate regularization generalized best on this dataset.
---

## Assumptions

- Neurons per hidden layer is fixed at **12** across all models.
- **Loss curve** is used for model history plots, as `MLPClassifier` exposes `loss_curve_` directly.
- An **80/20 train/test split** is used with `random_state=42` for reproducibility.
- `MLPClassifier` parameters used: `hidden_layer_sizes`, `activation`, `learning_rate_init`, `max_iter`, `alpha`, `random_state=42`. No early stopping is applied.
- Dataset ID `186` from `ucimlrepo` contains both red and white wine samples; both are used to increase sample size.
- The `color` column (red/white indicator) is retained as a feature after label encoding.