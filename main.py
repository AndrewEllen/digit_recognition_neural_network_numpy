import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display

# https://www.youtube.com/watch?v=w8yWXqWQYmU
# Building a neural network from scratch

data = pd.read_csv("training_data/train.csv")

#display(data.head())

# Loading in the data into a numpy array. The data is the 28x28 pixel data for the numbers. 784 rows.
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

# Initialising the parameters
def init_params():
  # Generating a random array with dimensions 10x784
  # Generates random values between 0 and 1
  W1 = np.random.randn(10, 784)
  # In tutorial he subtracts 0.5 from the below line. He also does the above line then says that breaks it.
  b1 = np.random.randn(10, 1)

  W2 = np.random.randn(10, 10)

  b2 = np.random.randn(10, 1)
  
  return W1, b1, W2, b2


# Function to handle rectified linear unit
def ReLU(Z):
  return np.maximum(Z, 0)


# Function to handle softmax
def softmax(Z):
  return np.exp(Z) / sum(np.exp(Z))


# Function to handle forward propogation
def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2


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
def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


# Function to update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1
  W2 = W2 - alpha * dW2
  b2 = b2 - alpha * db2
  return W1, b1, W2, b2


# Function to handle getting the prediction value
def get_predictions(A2):
  return np.argmax(A2, 0)


# Function to handle
def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y) / Y.size


# Function to handle gradient descent
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
  

# Function to make predictions
def make_predictions(X, W1, b1, W2, b2):
   _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
   predictions = get_predictions(A2)
   return predictions
   
# Function to test predictions
def test_prediction(index, W1, b1, W2, b2):
   current_image = X_train[:, index, None]
   prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
   label = Y_train[index]
   print("Prediction: ", prediction)
   print("Label ", label)

   current_image = current_image.reshape((28,28)) * 255
   plt.gray()
   plt.imshow(current_image, interpolation="nearest")
   plt.show()   
   
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1500, 0.1)
test_prediction(5, W1, b1, W2, b2)