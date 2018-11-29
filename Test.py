import numpy as np
import timeit
np.random.seed(0)

cols = 4
rows = 5
x = np.random.uniform(1, 4, (rows, cols))


class Activations:

    def __init__(self):
        self.value = 0
        self.back = None

    def sigmoid(self, z, back=False):
        sig = 1 / (1 + np.exp(-z))
        if back is False:
            return sig
        else:
            return sig*(1-sig)

    def relu(self,z):
        return np.maximum(z, 0)

    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z))

    def none(self,z):
        return(z)


def activate(z, func):
        switcher = {
                  'relu': activ.relu,
                  'sigmoid': activ.sigmoid,
                  'softmax': activ.softmax,
                  'none': activ.none
        }
        activation_choosen = switcher.get(func, 'invalid activation function')
        return activation_choosen(z)


activ = Activations()
z = np.array([1, 1, 1, 1, 1])
temp = activate(z, 'relu')
print(temp)

start = timeit.default_timer()
for k in range(3000):
    i = k

TempDNN = [{"input": 2, "output": 4, "nonlinear": "relu", "regularization": "l1"},
           {"input": 4, "output": 5, "nonlinear": "sigmoid", "regularization": "l1"},
           {"input": 5, "output": 1, "nonlinear": "softmax", "regularization": "l1"}]

print(TempDNN[1][ "regularization"])

stop = timeit.default_timer()

print('Time: ', stop - start)