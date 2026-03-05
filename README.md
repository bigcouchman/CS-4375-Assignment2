# CS-4375 Assignment 2 — Neural Networks

## Dataset

**Wine Quality (Red & White)**
UCI ML Repository ID: `186`
URL: https://archive.ics.uci.edu/dataset/186/wine+quality

Fetched programmatically using the `ucimlrepo` package

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

| Output | Description |
|---|---|
| `results.csv` | Hyperparameters, train/test accuracy, train/test MSE for all 48 models |
| `model_history_all.png` | Loss vs. epoch for all 48 models on a single figure |
| `model_history_by_activation.png` | Loss vs. epoch split into 3 subplots by activation function |

---