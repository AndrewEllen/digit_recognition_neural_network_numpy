import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display

# https://www.youtube.com/watch?v=w8yWXqWQYmU
# Building a neural network from scratch

data = pd.read_csv("training_data/train.csv")

display(data.head())

# Loading in the data into a numpy array. The data is the 28x28 pixel data for the numbers. 784 rows.
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


# Initialising the parameters
def init_params():

  # Generating a random array with dimensions 10x784
  # Generates random values between 0 and 1
  W1 = np.random.randn(10, 784)
  # In tutorial he subtracts 0.5 from the below line. He also does the above line then says that breaks it.
  b1 = np.random.randn(10, 1)

  W2 = np.random.randn(10, 784)
  # In tutorial he subtracts 0.5 from the below line. He also does the above line then says that breaks it.
  b2 = np.random.randn(10, 1)
  
  return W1, b1, W2, b2


# Function to handle rectified linear unit
def ReLU(Z):
  return np.maximum(0, Z)


# Function to handle softmax
def softmax(Z):
  return np.exp(Z) / np.sum(np.exp(Z))


# Function to handle forward propogation
def forward_prop(W1, b1, W2, b2, X):

  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(A1)


# Function to handle One hot encoding. This encodes categorical data into numerical ones.
def one_hot(Y):
  
  one_hot_Y = np.zeros((Y.size, Y.max() +1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y


# Function to handle derivitave of ReLU
def deriv_ReLU(Z):
  
  # derivitave of ReLU will be either 0 or 1 because the derivative is the gradient. When converting a bool to a number True = 1 and False = 0
  return Z > 0


# Function to handle backwards propogation
def back_prop(Z1, A1, Z2, A2, W2, Y):

  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2, 2)
  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(dZ2, 2)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ2, 2)
  return dW1, db1, dW2, db2


# Function to update parameters

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1
  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * db2
  return W1, b1, W2, b2