import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request
from mydnn import MyDNN
from mydnn import Activations
from mydnn import *
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------- Generate synthetic data set  ---------------------------------------

# raw_data = np.genfromtxt('/Users/Ido/Documents/MATLAB/Valerann/TU/DataSet_10-12-2018.csv', delimiter=',')
raw_data = np.genfromtxt('/Users/Ido/Documents/MATLAB/Valerann/TU/DataSet_13-12-2018.csv', delimiter=',')

test_end = 4800
val_start = 4800
val_end = 5200

np.random.shuffle(raw_data)

X_ = raw_data[0:test_end, 0:600] - 0.5
X_val = raw_data[val_start:val_end, 0:600] - 0.5
X_test = raw_data[val_start:val_end, 0:600] - 0.5

y_ = raw_data[0:test_end, 600].astype(int)
y_val = raw_data[val_start:val_end, 600].astype(int)
y_test = raw_data[val_start:val_end, 600].astype(int)

train_mean = find_mean(y_)
train_var = np.absolute(np.max(y_)-np.min(y_))

# y_ = (convert_vector_to_matrix(y_ - train_mean)) / train_var
# y_val = convert_vector_to_matrix(y_val - train_mean) / train_var

# y_test = convert_vector_to_matrix(y_test - train_mean) / train_var
y_ = (convert_vector_to_matrix(y_)) / train_var
y_val = convert_vector_to_matrix(y_val) / train_var
y_test = convert_vector_to_matrix(y_test) / train_var

# -------------------------------- Define network architecture --------------------------------------

DNN_count = [{"input": 600, "output": 128, "nonlinear": "relu", "regularization": "l2"},
             {"input": 128, "output": 64, "nonlinear": "sigmoid", "regularization": "l2"},
             {"input": 64, "output": 64, "nonlinear": "relu", "regularization": "l2"},
             {"input": 64, "output": 32, "nonlinear": "relu", "regularization": "l2"},
             {"input": 32, "output": 1, "nonlinear": "none", "regularization": "l2"}]


# DNN_count = [{"input": 600, "output": 128, "nonlinear": "relu", "regularization": "l1"},
#              {"input": 128, "output": 23, "nonlinear": "softmax", "regularization": "l1"}]

Loss = 'MSE'

weight_decay = 0.0000005

# ---------------------------------------- Init DNN -------------------------------------------------


DNN = MyDNN(DNN_count, Loss, weight_decay)


# -----------------------------------------  Train --------------------------------------------------

batch_size = 100

[trained_params, history] = DNN.fit(X_, y_, epochs=1000, batch_size=batch_size, learning_rate=1, learning_rate_decay=0.4, decay_rate=5, min_lr=0.0001, x_val=X_val, y_val=y_val)


# -------------------------------  Print loss curve -----------------------------------

# print_result(history['losses'], history['accu'], history['losses_val'], history['accu_val'], batch_size)

# # -----------------------------------  Evaluate Test set --------------------------------------------

[loss, accu, y_bar] = DNN.evaluate(X_test, y_test, None)
print_live(y_test, y_bar, -1)
plt.show(block=True)

lables_true = y_test * train_var
lables_bar = y_bar * train_var

# print(np.round(lables_bar), np.round(lables_bar).shape)
print(np.mean(lables_true-np.round(lables_bar)))
print(loss)
