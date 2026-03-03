#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class NeuralNet:
    def __init__(self, dataframe):
        # Initialize 
        self.raw_input = dataframe.copy()

    # Pre-processing
    def preprocess(self):
        dframe = self.raw_input.copy()

        # 1. Remove duplicate rows
        dframe.drop_duplicates(inplace=True)
        print(f"[Preprocess] Shape after dropping duplicates: {dframe.shape}")

        # 2. Separate features (X) and target (y)
        X = dframe.drop('quality', axis=1)
        y = dframe['quality']

        # 3. Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        # 4. Encode
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # 5. Standardization using Z-Score
        self.X_processed = (X - X.mean()) / X.std()
        self.y_processed = y

        print(f"[Preprocess] Features: {list(self.X_processed.columns)}")
        print(f"[Preprocess] Target classes: {sorted(self.y_processed.unique())}")

        return self.X_processed, self.y_processed

    # Train & Evaluate
    def train_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, self.y_processed,
            test_size=0.2, random_state=42
        )

        # Hyper-parameter grid
        # logistic = sigmoid 
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        max_iterations = [100, 200]          # epochs
        num_hidden_layers = [2, 3]
        neurons_per_layer = 12               # fixed neuron count per hidden layer
        alphas = [0.0001, 0.01]               # L2 regularization strength 

        # Train all models
        results = []
        model_curves = {}                    # label -> loss_curve

        total = (len(activations) * len(learning_rates) *
                 len(max_iterations) * len(num_hidden_layers) * len(alphas))
        count = 0

        for act in activations:
            for lr in learning_rates:
                for epochs in max_iterations:
                    for n_layers in num_hidden_layers:
                        for alpha in alphas:
                            count += 1
                            hidden = tuple([neurons_per_layer] * n_layers)
                            label = f"{act}_lr{lr}_ep{epochs}_hl{n_layers}_alpha{alpha}"

                            print("\n")
                            print(f"  Training model {count}/{total}: {label} …", end=" ")

                            mlp = MLPClassifier(
                                hidden_layer_sizes=hidden,
                                activation=act,
                                learning_rate_init=lr,
                                max_iter=epochs,
                                alpha=alpha,
                                random_state=42
                            )
                            mlp.fit(X_train, y_train)

                            # Predictions
                            pred_train = mlp.predict(X_train)
                            pred_test  = mlp.predict(X_test)

                            # Metrics
                            acc_train = mlp.score(X_train, y_train)
                            acc_test  = mlp.score(X_test, y_test)
                            mse_train = mean_squared_error(y_train, pred_train)
                            mse_test  = mean_squared_error(y_test, pred_test)

                            results.append({
                                "Model": label,
                                "Activation": act,
                                "Learning Rate": lr,
                                "Epochs": epochs,
                                "Hidden Layers": n_layers,
                                "Train Accuracy": round(acc_train, 4),
                                "Test Accuracy": round(acc_test, 4),
                                "Train MSE": round(mse_train, 4),
                                "Test MSE": round(mse_test, 4),
                            })
                            model_curves[label] = mlp.loss_curve_
                            print("done")

        # Results Table
        results_df = pd.DataFrame(results)
        print("\nRESULTS TABLE\n")
        print(results_df.to_string(index=False))

        # Model History Plots (loss vs. epoch)
        # Split into 3 sub-plots
        fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
        activation_labels = {a: [] for a in activations}

        for label, curve in model_curves.items():
            act = label.split("_")[0]
            activation_labels[act].append((label, curve))

        for ax, act in zip(axes, activations):
            for label, curve in activation_labels[act]:
                ax.plot(curve, label=label)
            ax.set_title(f"Activation: {act}", fontsize=13)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Model Loss Curves (Loss vs. Epoch) — All Hyperparameter Combinations",
                      fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig("model_history_by_activation.png", dpi=150, bbox_inches="tight")
        print("\n[Plot] Saved: model_history_by_activation.png")

        # 24 curves
        plt.figure(figsize=(16, 8))
        for label, curve in model_curves.items():
            plt.plot(curve, label=label)
        plt.title("Model Loss Curves — All Models", fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize=6, loc="upper right", ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("model_history_all.png", dpi=150, bbox_inches="tight")
        print("[Plot] Saved: model_history_all.png")

        # Summary & Analysis 
        print("\nSUMMARY\n")
        best_test = results_df.loc[results_df["Test Accuracy"].idxmax()]
        worst_test = results_df.loc[results_df["Test Accuracy"].idxmin()]
        print(f"Best  test accuracy : {best_test['Test Accuracy']:.4f}  — {best_test['Model']}")
        print(f"Worst test accuracy : {worst_test['Test Accuracy']:.4f}  — {worst_test['Model']}")

        for act in activations:
            subset = results_df[results_df["Activation"] == act]
            avg_acc = subset["Test Accuracy"].mean()
            print(f"  Avg test accuracy for '{act}': {avg_acc:.4f}")

        print("\nConclusion: See README.md for a detailed analysis of results.")

        return results_df


if __name__ == "__main__":
    # Fetch Wine Quality dataset from UCI ML Repo (Red & White)
    repo = fetch_ucirepo(id=186)
    df = repo.data.original.copy()
    neural_network = NeuralNet(df)
    neural_network.preprocess()
    neural_network.train_evaluate()
