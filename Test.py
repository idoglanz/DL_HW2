import matplotlib.pyplot as plt
import numpy as np
import pickle, gzip, urllib.request, json


# -------------------------------------- Activation class -----------------------------------------

class Activations:

    def __init__(self):
        self.value = 0
        self.back = None

    def sigmoid(self, z, back=False):
        print(z)
        sig = 1 / (1 + np.exp(-z))
        print(sig)
        if back is False:
            return sig
        else:
            return sig * (1-sig)

    def relu(self, z, back=False):
        if back is False:
            return np.maximum(z, 0)

        else:
            new_z = np.zeros((z.shape[0], z.shape[1]))
            new_z[z > 0] = 1

            return new_z

    def softmax(self, z, back=False):

        # a = np.exp(z)
        # b = (np.exp(z).sum(axis=1, keepdims=True))
        #
        # temp = np.divide(a, b)
        temp = np.exp(z) / (np.exp(z).sum(axis=1, keepdims=True))
        if back is False:
            return temp
        else:
            return temp*(np.eye(z.shape[0], z.shape[1])-temp)

    def none(self,z, back=False):
        if back is False:
            return z
        else:
            return 1


# ------------------------------------------ functions ------------------------------------------


def activate(z, func, back=False):
        switcher = {
                  'relu': activ.relu,
                  'sigmoid': activ.sigmoid,
                  'softmax': activ.softmax,
                  'none': activ.none
        }
        activation_chosen = switcher.get(func, 'invalid activation function')
        if back is False:
            return activation_chosen(z)
        else:
            return activation_chosen(z, True)


# # ############################################################################################################
# #
# #
# # def find_mean(data_set):
# #     train_mean = np.mean(data_set, axis=0)
# #     return train_mean
# #
# # ############################################################################################################
# #
# #
# # def preprocess(data_set, train_mean):
# #     data_set_processed = data_set - train_mean
# #     return data_set_processed
# #
# # ############################################################################################################
# #
# #
# # def class_process_old(y):
# #     y = [(1 if (y[i]%2 == 0) else 0) for i in range(len(y))]
# #     y = np.array(y)
# #     return y
# #
# #
# # def class_process_mnist(y, classes):
# #
# #     y_new = np.zeros((len(y), classes))
# #     row = np.linspace(0, y.shape[0]-1, y.shape[0], dtype='int')
# #     col = y
# #     y_new[row, col] = 1
# #
# #     return y_new
# #
# #
# # ############################################################################################################
# #
# #
# # def convert_vector_to_matrix(y):
# #     y = y[:, np.newaxis]
# #     return y
# #
# # # ------------------------------------------ Main ------------------------------------------
# #
# # # Load data-set and preprocess (normalize)
# #
# # # data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# # # urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
# # with gzip.open('mnist.pkl.gz', 'rb') as f:
# #     train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
# #
# # train_mean = find_mean(train_set[0])
# # X_ = preprocess(train_set[0], train_mean)
# # X_val = preprocess(valid_set[0], train_mean)
# # X_test = preprocess(test_set[0], train_mean)
# # y_ = convert_vector_to_matrix(class_process_mnist(train_set[1], 10))
# # y_val = convert_vector_to_matrix(class_process_mnist(valid_set[1], 10))
# # y_test = convert_vector_to_matrix(class_process_mnist(test_set[1], 10))
# #
# #
# # print(y_[3040, :])
# # print(train_set[1][3040])
# #

# X = np.array([[1, 2, 3], [10, 3, 4], [1, 2, 3], [1, 2, 3]])
# y = np.array([[1], [2], [3], [4]])
#
#
# activ = Activations()
# print(X)
# print(activate(X, 'sigmoid'))

temp = np.linspace(-2, 2, 1000)

[xx, yy] = np.meshgrid(temp, temp) #.reshape(2,-1).T

xy = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
print(xy)
print(xy.shape)
