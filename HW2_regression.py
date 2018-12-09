import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request
from mydnn import MyDNN
from mydnn import Activations
from mydnn import *
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------- Generate synthetic data set  ---------------------------------------

# f(x) = x1 * exp(-x1^2 -x2^2)


m = 1000

X_ = np.array([np.random.uniform(-2, 2, m), np.random.uniform(-2, 2, m)]).T

X_val = np.array([np.random.uniform(-2, 2, int(m/2)), np.random.uniform(-2, 2, int(m/2))]).T

X_test = np.array([np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000)]).T

y_ = X_[:, 0] + np.exp(-X_[:, 0]**2 - X_[:, 1]**2)
y_val = X_val[:, 0] + np.exp(-X_val[:, 0]**2 - X_val[:, 1]**2)
y_test = X_test[:, 0] + np.exp(-X_test[:, 0]**2 - X_test[:, 1]**2)

y_ = convert_vector_to_matrix(y_)
y_val = convert_vector_to_matrix(y_val)
y_test = convert_vector_to_matrix(y_test)

# -------------------------------- Define network architecture --------------------------------------

DNN_reg = [{"input": 2, "output": 200, "nonlinear": "relu", "regularization": "l2"},
            {"input": 200, "output": 100, "nonlinear": "sigmoid", "regularization": "l1"},
            {"input": 100, "output": 1, "nonlinear": "none", "regularization": "l2"}]


Loss = 'MSE'

weight_decay = 0.01

# ---------------------------------------- Init DNN -------------------------------------------------


DNN = MyDNN(DNN_reg, Loss, weight_decay)


# -----------------------------------------  Train --------------------------------------------------

batch_size = 100

[trained_params, history] = DNN.fit(X_, y_, epochs=1000, batch_size=batch_size, learning_rate=0.1, learning_rate_decay=0.5, decay_rate=5, min_lr=0.001, x_val=X_val, y_val=y_val)


# -------------------------------  Print loss curve -----------------------------------

print_result(history['losses'], None, history['losses_val'], None, batch_size)


# # -----------------------------------  Evaluate Test set --------------------------------------------


[loss, accu, y_bar] = DNN.evaluate(X_test, y_test, None)

print(loss, accu)

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test[:, 0], X_test[:, 1], y_test, 'b')
ax.scatter(X_test[:, 0], X_test[:, 1], y_bar, 'r')

# plt.show()
plt.ion()
plt.pause(0.1)
plt.show(block=True)
