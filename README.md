# Planar Data Classification with One Hidden Layer 🌟

This repository contains a Python implementation of a simple neural network for classifying planar data using logistic regression and a basic neural network architecture.

## Table of Contents 📚
1. [Packages](#packages)
2. [Dataset](#dataset)
3. [Simple Logistic Regression](#simple-logistic-regression)
4. [Neural Network Model](#neural-network-model)
   - [Defining the Neural Network Structure](#defining-the-neural-network-structure)
   - [Initialize the Model's Parameters](#initialize-the-models-parameters)
   - [Forward Propagation](#forward-propagation)
   - [Compute Cost](#compute-cost)
   - [Backward Propagation](#backward-propagation)
5. [Credits](#credits)

## Packages 🛠️

To get started, you'll need to install the following packages:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases_v2 import *
```

### Ensure Inline Plotting for Jupyter Notebooks
```python
%matplotlib inline
```

### Set Random Seed for Reproducibility
```python
np.random.seed(1)
```

## Dataset 📊

Load the planar dataset and visualize it:

```python
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()
```

Shapes of the datasets:
```python
shape_X = X.shape  # Shape of input features
shape_Y = Y.shape  # Shape of labels
m = shape_X[1]  # Number of training examples
print(f'The shape of X is: {shape_X}')
print(f'The shape of Y is: {shape_Y}')
print(f'I have {m} training examples!')
```

## Simple Logistic Regression 🤖

Train a logistic regression classifier and evaluate its accuracy:

```python
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV()
clf.fit(X.T, Y.ravel())

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)
accuracy = np.mean(LR_predictions == Y.ravel()) * 100
print(f'Accuracy of logistic regression: {accuracy:.2f}%')
```

## Neural Network Model 🧠

### Defining the Neural Network Structure

Function to determine sizes of layers:
```python
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y
```

### Initialize the Model's Parameters

Function to initialize parameters:
```python
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters
```

### Forward Propagation

Function to implement forward propagation:
```python
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache
```

### Compute Cost

Function to compute the cross-entropy cost:
```python
def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = (-1 / m) * np.sum(logprobs)
    cost = float(np.squeeze(cost))
    return cost
```

### Backward Propagation

Function to implement backward propagation (to be completed):
```python
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Compute gradients...
```

## Credits 🙏

This project is inspired by the **Deep Learning Specialization** from [DeepLearning.AI](https://www.deeplearning.ai/courses/deep-learning-specialization/). Thank you for the amazing resources and learning materials! 🎓
