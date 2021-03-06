import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request
from mydnn import MyDNN
from mydnn import Activations
from mydnn import *


# THIS IS THE SCRIPT USED FOR THE ARCHITECTURE TESTING - THE PARAMETERS WERE ALTERED TO MATCH THE DIFFERENT TESTS


# -------------------------------- Load MNIST and preprocess (normalize) ---------------------------------------

# data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_mean = find_mean(train_set[0])

X_ = preprocess(train_set[0], train_mean)
X_val = preprocess(valid_set[0], train_mean)
X_test = preprocess(test_set[0], train_mean)

y_ = class_process_mnist(train_set[1], 10)
y_val = class_process_mnist(valid_set[1], 10)
y_test = class_process_mnist(test_set[1], 10)


# -------------------------------- Define network architecture --------------------------------------

DNN_q1 = [{"input": 784, "output": 512, "nonlinear": "relu", "regularization": "l2"},
          {"input": 512, "output": 10, "nonlinear": "softmax", "regularization": "l2"}]

# DNN_q1 = [{"input": 784, "output": 512, "nonlinear": "relu", "regularization": "l2"},
#           {"input": 512, "output": 80, "nonlinear": "sigmoid", "regularization": "l2"},
#           {"input": 80, "output": 10, "nonlinear": "softmax", "regularization": "l2"}]

Loss = 'cross_entropy'

weight_decay =  5*10**-5

# ---------------------------------------- Init DNN -------------------------------------------------

DNN = MyDNN(DNN_q1, Loss, weight_decay)

# -----------------------------------------  Train --------------------------------------------------

batch_size = 1024

start = timeit.default_timer()

[trained_params, history] = DNN.fit(X_, y_, epochs=20, batch_size=batch_size, learning_rate=1, learning_rate_decay=1,
                                    decay_rate=1, min_lr=0.0, x_val=None, y_val=None)

stop = timeit.default_timer()

# -----------------------------------  Evaluate Test set --------------------------------------------


[loss, accu, y_bar] = DNN.evaluate(X_test, y_test, None)

print(["Runtime = " + str(stop-start) + ', Test set loss = ' + str(loss) + ', Test set accuracy = ' + str(accu)])

