#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
# from ucimlrepo import fetch_ucirepo

class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, sep = ';')

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        dframe = self.raw_input.fillna(self.raw_input.mean())
        X = dframe.drop('quality', axis=1)
        y = dframe['quality']

        self.X_processed = (X - X.mean()) / X.std()
        self.y_processed = y

        return self.X_processed, self.y_processed

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_processed, self.y_processed, test_size = 0.2, random_state = 42)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        results = []
        plt.figure(figsize=(15, 10))
        for i in activations:
            for j in learning_rate:
                for k in max_iterations:
                    for l in num_hidden_layers:
                        hidden_layers = tuple([32] * l)
                        mlpClass = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=i, learning_rate_init=j, max_iter=k, random_state=1)
                        mlpClass.fit(X_train, y_train)

                        predict_train = mlpClass.predict(X_train)
                        predict_test = mlpClass.predict(X_test)

                        accur_train = mlpClass.score(X_train, y_train)
                        accur_test = mlpClass.score(X_test, y_test)
                        mse_train = mean_squared_error(y_train, predict_train)
                        mse_test = mean_squared_error(y_test, predict_test)

                        title = f"{i}_lr{j}_ep{k}_lay{l}"
                        results.append({
                            "Label": title,
                            "Training Accuracy": round(accur_train, 4),
                            "Test Accuracy": round(accur_test, 4),
                            "Training MSE": round(mse_train, 4),
                            "Test MSE": round(mse_test, 4),
                        })
                        plt.plot(mlpClass.loss_curve_, label=title)

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
            # in a same figure.
        plt.title("NN Training History: All Models")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.tight_layout()
        plt.show()
        dframe_results = pd.DataFrame(results)
        print("\n--- Result table ---")
        print(dframe_results.to_string(index=False))

if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    neural_network = NeuralNet(url) # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
