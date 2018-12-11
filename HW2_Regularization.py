import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request
from mydnn import MyDNN
from mydnn import Activations
from mydnn import *

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

DNN_q1 = [{"input": 784, "output": 128, "nonlinear": "relu", "regularization": "None"},
          {"input": 128, "output": 10, "nonlinear": "softmax", "regularization": "None"}]

Loss = 'cross_entropy'

loss = []
accu = []

weight_decay_vector = [5*10**-7, 5*10**-6, 1*10**-5, 5*10**-5, 1*10**-4, 5*10**-4]

for k in range(len(weight_decay_vector)):

    weight_decay = weight_decay_vector[k]

    # ---------------------------------------- Init DNN -------------------------------------------------

    DNN = MyDNN(DNN_q1, Loss, weight_decay)

    # -----------------------------------------  Train --------------------------------------------------

    batch_size = 1024

    [trained_params, history] = DNN.fit(X_, y_, epochs=10, batch_size=batch_size, learning_rate=1, learning_rate_decay=1, decay_rate=1, min_lr=0.0, x_val=None, y_val=None)

    # -------------------------------  Print loss and accuracy curves -----------------------------------

    # print_result(loss=history['losses'], accu=history['accus'], loss_val=history['losses_val'], accu_val=history['accus_val'], batch_size=batch_size)

    # -----------------------------------  Evaluate Test set --------------------------------------------

    [loss_temp, accu_temp, y_bar] = DNN.evaluate(X_val, y_val, None)
    loss.append(loss_temp)
    accu.append(accu_temp)

    print(["Weight decay = " + str(weight_decay_vector[k]) + ', loss = ' + str(loss_temp) + ', accuracy = ' + str(accu_temp)])


fig1 = plt.figure(1)

plt.plot(weight_decay_vector, loss, 'ob')

plt.ylabel('Loss')
plt.xlabel('Weight decay value')
plt.title(['Loss VS weight decay - None'])
plt.grid()
fig1.show()

fig2 = plt.figure(2)

plt.plot(weight_decay_vector, accu, 'or')

plt.ylabel('Accuracy')
plt.xlabel('Weight decay value')
plt.title(['Accuracy VS weight decay - None'])
plt.grid()
fig2.show()

plt.ion()
plt.pause(0.1)
plt.show(block=True)

print(loss)
print(accu)



