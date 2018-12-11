import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request
from mydnn import MyDNN
from mydnn import Activations
from mydnn import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

# -------------------------------- Generate synthetic data set  ---------------------------------------

# f(x) = x1 * exp(-x1^2 -x2^2)


m = 1000

X_ = np.array([np.random.uniform(-2, 2, m), np.random.uniform(-2, 2, m)]).T

X_val = np.array([np.random.uniform(-2, 2, int(m/2)), np.random.uniform(-2, 2, int(m/2))]).T


[xx, yy] = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

X_test = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)


y_ = X_[:, 0] + np.exp(-X_[:, 0]**2 - X_[:, 1]**2)
y_val = X_val[:, 0] + np.exp(-X_val[:, 0]**2 - X_val[:, 1]**2)
y_test = X_test[:, 0] + np.exp(-X_test[:, 0]**2 - X_test[:, 1]**2)

y_ = convert_vector_to_matrix(y_)
y_val = convert_vector_to_matrix(y_val)
y_test = convert_vector_to_matrix(y_test)

# -------------------------------- Define network architecture --------------------------------------

DNN_reg = [{"input": 2, "output": 128, "nonlinear": "relu", "regularization": "l2"},
            {"input": 128, "output": 64, "nonlinear": "sigmoid", "regularization": "l2"},
            {"input": 64, "output": 1, "nonlinear": "none", "regularization": "l2"}]


# DNN_reg = [{"input": 2, "output": 400, "nonlinear": "relu", "regularization": "l2"},
#             {"input": 400, "output": 1, "nonlinear": "none", "regularization": "l2"}]


Loss = 'MSE'

loss = []

weight_decay_vector = [5*10**-7, 5*10**-6, 1*10**-5, 5*10**-5, 1*10**-4, 5*10**-4, 1*10**-3, 5*10**-3, 1*10**-2, 5*10**-2]


# for k in range(len(weight_decay_vector)):
for k in range(1):

    weight_decay = weight_decay_vector[k]

    # ---------------------------------------- Init DNN -------------------------------------------------


    DNN = MyDNN(DNN_reg, Loss, weight_decay)


    # -----------------------------------------  Train --------------------------------------------------

    batch_size = 100

    [trained_params, history] = DNN.fit(X_, y_, epochs=200, batch_size=batch_size, learning_rate=0.1, learning_rate_decay=1, decay_rate=1, min_lr=0.01, x_val=X_val, y_val=y_val)


    # -------------------------------  Print loss curve -----------------------------------

    print_result(history['losses'], None, history['losses_val'], None, batch_size)


    # # -----------------------------------  Evaluate Test set --------------------------------------------


    # [loss, accu, y_bar] = DNN.evaluate(X_test, y_test, None)

    [loss_temp, accu_temp, y_bar] = DNN.evaluate(X_test, y_test, None)
    loss.append(float(loss_temp))

    print(["Weight decay = " + str(weight_decay_vector[k]) + ', loss = ' + str(loss_temp) + ', accuracy = ' + str(accu_temp)])

# fig1 = plt.figure(1)
# print(loss, len(loss), len(weight_decay_vector))
# plt.plot(weight_decay_vector, loss, 'ob')
#
# plt.ylabel('Loss')
# plt.xlabel('Weight decay value')
# plt.title(['Loss VS weight decay'])
# plt.grid()
# fig1.show()


fig = plt.figure(3)

ax = plt.axes(projection='3d')
ax.plot_surface(xx, yy, y_test.reshape(-1, 100), rstride=1, cstride=1, edgecolor='none')
ax.plot_surface(xx, yy, y_bar.reshape(-1, 100), rstride=1, cstride=1, edgecolor='none')

ax.set_title('Function Surface Plot')

plt.show()
plt.ion()
plt.pause(0.1)
plt.show(block=True)
