# CS 4375 Assignment 2 Part 2 By Nguyen Do (npd220001) and Casey Nguyen (cxn220034)
# This assignment implements a neural network model and fine tuning parameters for each activation
# on the UCI wine quality dataset. Question answered in report.md

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

class NeuralNet:
    def __init__(self, d_id = 186):
        # Get Red and White wine data from UCI dataset, get features and target
        dataset = fetch_ucirepo(id = d_id)
        X = dataset.data.features
        y = dataset.data.targets

        self.raw_input = pd.concat([X, y], axis=1)
        self.target_name = y.columns[0]

    # Preprocess data set with standardization and label encoding
    def preprocess(self):
        # Handle null and duplicates, helping on stability
        dframe = self.raw_input.fillna(self.raw_input.mean())
        dframe.drop_duplicates(inplace=True)

        
        X = dframe.drop(self.target_name, axis=1)
        y = dframe[self.target_name]

        for c in X.select_dtypes(include=['object']).columns:
            label = LabelEncoder()
            X[c] = label.fit_transform(X[c])

        # Standardizing data by putting it to the same scale
        self.X_processed = (X - X.mean()) / X.std()
        self.y_processed = y

        return self.X_processed, self.y_processed

    # Train and evaluate all possible models, keep track of training/test accurcy and errors, and a history curve plot

    def train_evaluate(self):
        # Train test split (80/20), create directories to save logs and graphs
        X_train, X_test, y_train, y_test = train_test_split(self.X_processed, self.y_processed, test_size = 0.2, random_state = 42)
        os.makedirs("plots", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        alphas = [0.0001, 0.01]

        # Store metrics in result array and create figure grid for graphs
        results = []
        fig, axes = plt.subplots(1, 3, figsize=(20,6), sharey=True)

        # Fine tune parameters
        for element, i in enumerate(activations):
            ax = axes[element]
            for j in learning_rate:
                for k in max_iterations:
                    for l in num_hidden_layers:
                        for alp in alphas:

                            # Set a low number of neurons to avoid overfitting
                            hidden_layers = tuple([12] * l)
                            mlpClass = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=i, learning_rate_init=j, max_iter=k, alpha = alp, random_state=42)

                            # Suppress warning
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                mlpClass.fit(X_train, y_train)

                            # Calculate metrics for performance evaluation
                            predict_train = mlpClass.predict(X_train)
                            predict_test = mlpClass.predict(X_test)

                            accur_train = mlpClass.score(X_train, y_train)
                            accur_test = mlpClass.score(X_test, y_test)
                            mse_train = mean_squared_error(y_train, predict_train)
                            mse_test = mean_squared_error(y_test, predict_test)

                            title = f"{i}_lr{j}_ep{k}_lay{l}_alpha{alp}"
                            results.append({
                                "Activation": i,
                                "Label": title,
                                "Alpha": alp,
                                "Training Accuracy": round(accur_train, 4),
                                "Test Accuracy": round(accur_test, 4),
                                "Training Loss": round(mse_train, 4),
                                "Test Loss": round(mse_test, 4),
                                "Epochs": len(mlpClass.loss_curve_),
                                "Loss History": mlpClass.loss_curve_
                            })

                            # Plotting loss over time
                            ax.plot(mlpClass.loss_curve_, label=title)

            # Create a plot for each activation function, modeling loss history for every respective activation model
            ax.set_title(f"NN Training History: {i}")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("MSE")
            ax.legend(fontsize='xx-small', loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.figure(fig.number)
        plt.suptitle("Neural Network Training History", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Store graphs and logs
        plotting = os.path.join("plots", "nn_training_history.png")
        plt.savefig(plotting, dpi=300)
        
        # Create a general history loss plot for all 3 activations (48 models)
        plt.figure(figsize=(12, 8))
        for r in results:
            plt.plot(r["Loss History"], label = r["Label"], linewidth=0.7, alpha=0.6)
        
        plt.title("Neural Network Training History for all Models", fontsize = 16)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 'xx-small', ncol=2)
        plt.grid(True, alpha=0.3)

        big_plotting = os.path.join("plots", "all_models_history.png")
        plt.savefig(big_plotting)
        
        # Gather metrics via a dataframe to display in log
        dframe_results = pd.DataFrame(results)
        accur_avg_train = dframe_results.groupby('Activation')['Training Accuracy'].mean()
        accur_avg_test = dframe_results.groupby('Activation')['Test Accuracy'].mean()
        best_train = dframe_results.loc[dframe_results['Training Accuracy'].idxmax()]
        worst_train = dframe_results.loc[dframe_results['Training Accuracy'].idxmin()]
        best_test = dframe_results.loc[dframe_results['Test Accuracy'].idxmax()]
        worst_test = dframe_results.loc[dframe_results['Test Accuracy'].idxmin()]
        
        # Get rid of Loss History as it messes up the table (need to plot only), store table in results.csv
        cleaned_dframe = dframe_results.drop(columns='Loss History')
        
        logging = os.path.join("logs", "results.csv")
        cleaned_dframe.to_csv(logging, index=False)

        # Create an additional file logging performance metrics
        metrics = os.path.join("logs", "metrics.txt")
        with open(metrics, "w") as f:
            f.write("Average Test Accuracy per Activation: ")
            f.write(accur_avg_test.to_string())
            f.write(f"\nBest Accuracy: {best_test['Label']} | Accuracy: {best_test['Test Accuracy']}")
            f.write(f"\nWorst Accuracy: {worst_test['Label']} | Accuracy: {worst_test['Test Accuracy']}")
            f.write("Average Training Accuracy per Activation: ")
            f.write(accur_avg_train.to_string())
            f.write(f"\nBest Accuracy: {best_train['Label']} | Accuracy: {best_train['Training Accuracy']}")
            f.write(f"\nWorst Accuracy: {worst_train['Label']} | Accuracy: {worst_train['Training Accuracy']}")

        # Print the results
        print("\n--- Result table ---")
        print(cleaned_dframe.to_string(index=False))

        print("Average Test Accuracy per Activation: ")
        print(accur_avg_test)
        print(f"Best Accuracy: {best_test['Label']} | Accuracy: {best_test['Test Accuracy']}")
        print(f"Worst Accuracy: {worst_test['Label']} | Accuracy: {worst_test['Test Accuracy']}")

        print("Average Training Accuracy per Activation: ")
        print(accur_avg_train)
        print(f"Best Accuracy: {best_train['Label']} | Accuracy: {best_train['Training Accuracy']}")
        print(f"Worst Accuracy: {worst_train['Label']} | Accuracy: {worst_train['Training Accuracy']}")

# Main to call neural network funcitons
if __name__ == "__main__":
    neural_network = NeuralNet(186) # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
