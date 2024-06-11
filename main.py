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

