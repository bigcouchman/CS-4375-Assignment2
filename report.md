# CS-4375 Assignment 2 - Report

## Hyper Parameters:
- Activation Function: logistic (sigmoid), tanh, relu
- Learning Rate: 0.01, 0.1
- Max Iterations: 100, 200 (epochs)
- Number of Hidden Layers: 2, 3
- Alpha (Regularization): 0.0001, 0.01
- Number of neurons: 12 (fixed)

## Results
### Performance Metrics
Metric | Model | Value
---|---|---|
Best Test Accuracy|Logistic, Learning Rate = 0.1, epochs = 100, layer = 2, alpha = 0.01| 0.5827
Worst Test Accuracy|ReLu, Learning Rate = 0.1, epochs = 100, layer = 2, alpha = 0.0001| 0.5263
Best Training Accuracy|Tanh, Learning Rate = 0.01, epochs = 200, layer = 3, alpha = 0.0001| 0.6244
Worst Training Accuracy|ReLu, Learning Rate = 0.1, epochs = 100, layer = 3, alpha = 0.01| 0.5362
- This is for 48 model combinations
### Average Accuracies by Activation Function
Activation | Training/Test | Average Accuracy
---|---|---|
Logistic|Training|0.5852
Logistic|Test|0.5639
Tanh|Training|0.5894
Tanh|Test|0.5501
ReLu|Training|0.5724
ReLu|Test|0.5467
- For metrics of all 48 model combinations, check results.csv

## Analysis

### Activation Function
- Logistic is the most reliable performer at with a 56.4% accuracy. Wine Quality in the dataset does not have great variance (scores concentrated
  at around 5 to 7), the structure of the Logistic function handles overlapping datapoints quite efficiently.
- ReLu is rather unstable because it has the lowest test accuracy at 52.6%. ReLu often overshoots the best weight when paired with a high learning rate.
- Tanh performs almost as well as Logistic at 55.0% accuracy, and while it has the highest training accuracy at 62.4%, the model is overfitting with the data.
- From reviewing the accuracy gaps between the activations, logistic has a smaller gap from training to test accuracies, showing a better generalized model

### Learning Rate
- Learning rate 0.01 is the most reliable because of the imbalance nature of the wine dataset, which forces the model to slowly learn and find optimal weights
- Learning rate 0.1 is not as reliable because the model can skip over optimal weight, which does not decrease the loss

### Epochs 
- For a slow learning rate of 0.01, it is best to pair with a higher epoch. Some models improved at 200 epochs.

### Hidden Layers
- The test model best performs with 2 layers while the training model best performs with 3. Adding more layers may help the model memorize but does not help with different dataset.

### Regularization (alpha)
- Alpha 0.01 performs better because it penalizes the model for having complex weights.

### Summary:
- While there are some outliers, most models perform at 52% to 58% accuracy. The wine dataset have scores mostly skewed at 5 to 7 (described above), making
  other minority scores harder to predict. However, a model comprised of logistic activation with a lower learning rate, 2 hidden layers, and medium regularization performs and generalizes the best.

## Assumptions
- Neurons are fixed at 12 across all models
- Loss curve is for model history plots, as MLPClassifer shows loss_curve_ directly
- Dataset ID is 186 from ucimlrepo (Wine Quality, Red and White)
