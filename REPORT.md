# CS-4375 Assignment 2 — Report

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

Full results table in `results.csv`.

---

## Analysis

### Activation Functions
**Logistic (sigmoid)** had the highest average test accuracy (55.92%) and produced the best model. **ReLU** ranked second (54.71%), while **tanh** ranked last (53.93%). All Activation Functions performed similarly. 

### Learning Rate
A learning rate of **0.01** outperformed **0.1**, especially for logistic and tanh activations. Higher learning rates increased the risk of overshooting the minima, for example the tanh function where the worst model (`tanh_lr0.1_ep100_hl2_alpha0.01`, 51.13%) was also a high learning rate. ReLU was not affected much.

### Number of Epochs (100 vs. 200)
Several models improved at 200 epochs. For example, `logistic_lr0.01_ep200_hl2_alpha0.01` (58.08%) outperforms its 100-epoch model (57.33%), confirming that 100 epochs was sometimes not enough for convergence at lr=0.01.

### Depth (2 vs. 3 Hidden Layers)
Results were mixed. For logistic at lr=0.01, 2 hidden layers slightly outperformed 3. For tanh and relu, 3 layers sometimes produced higher training accuracy but did not reliably improve test accuracy. This can suggest slight overfitting when exploring deeper in the network. 

### L2 Regularization (alpha)
The effect of alpha was small. In the logistic group, alpha=0.01 produced the best single model (58.08%), showing that a regularization penalty helped generalization on this diffcult dataset. Without regularization, some models reached 91% training accuracy but only ~50% test accuracy, indicating severe overfitting.

### Overfitting
Several tanh and relu models show a noticeable train/test accuracy gap (e.g., `tanh_lr0.01_ep200_hl3_alpha0.0001`: train 62.92%, test 54.42%). Logistic models showed more balanced gaps, leading to stronger generalization overall.

### Overall
All models fall in a ~51–58% accuracy range, showing the difficulty of the wine quality 7-class problem. The class distribution is heavily skewed toward quality scores 5 and 6, making minority classes (3, 4, 8, 9) hard to predict. Logistic activation with a low learning rate, 2 hidden layers, and moderate regularization generalized best on this dataset.

---

## Assumptions

- Neurons per hidden layer is fixed at **12** across all models.
- **Loss curve** is used for model history plots, as `MLPClassifier` exposes `loss_curve_` directly.
- An **80/20 train/test split** is used with `random_state=42` for reproducibility.
- No early stopping is applied — `max_iter` is the hard epoch budget.
- Dataset ID `186` from `ucimlrepo` (Red and White Wine).
- The `color` column (red/white indicator) is retained as a feature after label encoding.