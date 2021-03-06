import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request
from mydnn import MyDNN
from mydnn import Activations
from mydnn import *
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------- Generate synthetic data set  ---------------------------------------

raw_data = np.genfromtxt('/Users/Ido/Documents/MATLAB/Valerann/TU/DataSet_10-12-2018.csv', delimiter=',')

test_end = 2800
val_start = 2801
val_end = 3100

np.random.shuffle(raw_data)

X_ = raw_data[0:test_end, 0:600] - 0.5
X_val = raw_data[val_start:val_end, 0:600] - 0.5
X_test = raw_data[val_start:val_end, 0:600] - 0.5

# X_ = raw_data[0:2800, 0:600] - 0.5
# X_val = raw_data[2800:3000, 0:600] - 0.5
# X_test = raw_data[2800:3000, 0:600] - 0.5

# print(X_[1, :])
# print(X_.shape)

# y_ = raw_data[0:4000, 600].astype(int)
# y_val = raw_data[4001:4500, 600].astype(int)
# y_test = raw_data[4500:5160, 600].astype(int)

y_ = raw_data[0:test_end, 600].astype(int)
y_val = raw_data[val_start:val_end, 600].astype(int)
y_test = raw_data[val_start:val_end, 600].astype(int)

y_ = class_process_mnist(y_, 23)
y_val = class_process_mnist(y_val, 23)
y_test = class_process_mnist(y_test, 23)

# y_ = convert_vector_to_matrix(y_ - train_mean)
# y_val = convert_vector_to_matrix(y_val - train_mean)
# y_test = convert_vector_to_matrix(y_test - train_mean)


# -------------------------------- Define network architecture --------------------------------------

DNN_count = [{"input": 600, "output": 128, "nonlinear": "relu", "regularization": "l2"},
             {"input": 128, "output": 64, "nonlinear": "sigmoid", "regularization": "l2"},
             {"input": 64, "output": 23, "nonlinear": "softmax", "regularization": "l2"}]


# DNN_count = [{"input": 600, "output": 128, "nonlinear": "relu", "regularization": "l1"},
#              {"input": 128, "output": 23, "nonlinear": "softmax", "regularization": "l1"}]

Loss = 'cross_entropy'

weight_decay = 0.01

# ---------------------------------------- Init DNN -------------------------------------------------


DNN = MyDNN(DNN_count, Loss, weight_decay)


# -----------------------------------------  Train --------------------------------------------------

batch_size = 10

[trained_params, history] = DNN.fit(X_, y_, epochs=100, batch_size=batch_size, learning_rate=0.01, learning_rate_decay=1, decay_rate=1, min_lr=0.001, x_val=X_val, y_val=y_val)


# -------------------------------  Print loss curve -----------------------------------

# print_result(history['losses'], history['accu'], history['losses_val'], history['accu_val'], batch_size)


# # -----------------------------------  Evaluate Test set --------------------------------------------

[loss, accu, y_bar] = DNN.evaluate(X_test, y_test, None)
print(loss, accu)
print_output(np.argmax(y_test, axis=1), np.argmax(y_bar, axis=1), 0, blck=True)
