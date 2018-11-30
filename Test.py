import matplotlib.pyplot as plt
import numpy as np
import pickle, gzip, urllib.request, json


def convert_vector_to_matrix(y):
    y = y[:, np.newaxis]
    return y


def shuffle_data(x, y):
    m = x.shape[1]
    print(m)
    xy = np.concatenate((x, y), axis=1)
    np.random.shuffle(xy)
    print(xy)
    return xy[:, :-1], convert_vector_to_matrix(xy[:, m])


X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
y = np.array([[1], [2], [3], [4]])

print(X.shape)
print(y.shape)

[X_new, y_new] = shuffle_data(X, y)
print(y_new.shape)
print('')
print(X - X_new)
print(y - y_new)

w = np.zeros(5, 1)
print(w)