# CS-4375-Assignment2

## Dataset
### Wine Quality (Red and White) with UCI ML Repository (ID = 186): https://archive.ics.uci.edu/dataset/186/wine+quality

## Requirements
- Install dependencies via:
`pip install -r requirements.txt`
- Dependencies: numpy, panda, matplotlib, scikit-learn, ucimlrepo
- Specific libraries:
  - sklearn.model_selection.train_test_split: Splitting data to test and train dataset
  - sklearn.neural_network.MLPClassifier: Create a Multi Layer Perceptron as a Neural Network
  - sklearn.metrics: Tracking MSE as an additional metric (Accuracy is more important)
  - sklearn.exceptions: Error Suppression (Specifically Convergence Warnings)
  - sklearn.preprocessing.LabelEncoder: Encoding Categorical variables

### Program
- To run: python NeuralNet.py
- The code:
  - Load and preprocess dataset
  - Fine tune hyperparameters (noted in report.md)
  - Evaluate model on test data
  - Print results and generate log files and plot graphs

### File Contents and Structures
- logs/: Track hyper parameters for each model (all 48 possible parameter combinations) and metrics
- plots/: Track loss history of all 48 models and for each activation function
- CS4375_CoverPage.docx: Cover Page of the assignment
- NeuralNet.py: Code for the assignment
- README.md: Set up instructions for program
- report.md: Detailed report
- requirements.txt: To install needed dependencies
