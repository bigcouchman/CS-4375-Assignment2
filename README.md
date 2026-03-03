# CS-4375 Assignment 2 — Neural Networks

## Overview

This project trains and evaluates multiple neural network models on the **Wine Quality (Red)** dataset from the UCI Machine Learning Repository. It explores all combinations of the following hyper-parameters and produces performance metrics and loss-curve plots.

## Dataset

**Wine Quality — Red Wine**  
UCI ML Repository: <https://archive.ics.uci.edu/ml/datasets/Wine+Quality>  
Fetched programmatically using the `ucimlrepo` Python package (dataset ID: 186), filtered to red wine only.

The dataset contains 1,599 samples (1,359 after deduplication) with 11 physicochemical input features and one output feature (`quality`, an integer score from 3–8).

## Requirements

- Python 3.8+
- Packages listed in `requirements.txt`

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

### Output

- A **results table** printed to the console showing hyper-parameters, training/test accuracy, and training/test MSE for every model.
- **`model_history_all.png`** — Loss vs. epoch plot with all 24 models on a single figure.
- **`model_history_by_activation.png`** — Loss vs. epoch plots split by activation function.

## Hyper-Parameter Grid

| Parameter | Values |
|---|---|
| Activation Function | `logistic`, `tanh`, `relu` |
| Learning Rate | `0.01`, `0.1` |
| Max Iterations (Epochs) | `100`, `200` |
| Hidden Layers | `2`, `3` |

Total combinations: **3 × 2 × 2 × 2 = 24 models**

## Pre-processing Steps

1. **Duplicate removal** — Dropped exact duplicate rows (1,599 → 1,359).
2. **Missing value handling** — Filled numeric NaN values with the column mean.
3. **Categorical encoding** — Any non-numeric columns are label-encoded.
4. **Standardization** — Z-score normalization applied to all features.

## Results Summary

After training all 24 models on red wine data, the key findings are:

- **Best model:** `logistic_lr0.01_ep100_hl3` with **62.87% test accuracy**
- **Worst model:** `relu_lr0.01_ep200_hl2` with **48.90% test accuracy**
- **Logistic activation** performed best on average (avg test accuracy: **0.5551**), likely because the dataset is small and well-scaled, where logistic converges reliably without overfitting aggressively
- **ReLU** was second (avg test accuracy: **0.5492**), performing comparably to logistic in several configurations
- **Tanh** performed worst on average (avg test accuracy: **0.5345**) and showed severe overfitting — for example, `tanh_lr0.01_ep200_hl3` reached 99.91% train accuracy but only 52.94% test accuracy
- **Lower learning rate (0.01)** consistently produced better generalization than 0.1, which caused unstable or aggressive training
- **100 epochs often outperformed 200 epochs** — additional training without regularization led to overfitting rather than improved generalization
- **Several models with lr=0.1 plateaued** before 100 epochs, producing identical results at ep100 vs ep200 (e.g., both `tanh_lr0.1_ep100_hl2` and `tanh_lr0.1_ep200_hl2` show identical metrics)
- **3 hidden layers did not consistently outperform 2** — the dataset is relatively small and does not always benefit from the added depth

Overall, the dataset favors stable, conservative optimization. The logistic activation with a low learning rate and fewer epochs generalized best, while deeper or more aggressively trained models tended to overfit. The dataset type Wine Quality and dataset size also make it hard to get better test accuracy. We experimented added L2 regularization (alpha) and it didn't help much either.

## Assumptions

- Fixed 32 neurons per hidden layer across all models
- Used loss curve (not accuracy curve) for model history plots
- 80/20 train/test split with `random_state=42`
- No regularization applied (used sklearn default)
- Dataset ID 186 from `ucimlrepo` contains both red and white wine; code explicitly filters to red wine only and drops the `color` column before training