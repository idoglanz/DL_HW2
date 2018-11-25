import numpy as np
import matplotlib as plt

TempDNN = [{"input": 2, "output": 4, "nonlinear": "relu", "regularization": "l1"},
           {"input": 4, "output": 5, "nonlinear": "sigmoid", "regularization": "l1"},
           {"input": 5, "output": 1, "nonlinear": "softmax", "regularization": "l1"}]

Loss = 'MSE'
weight_decay = 0.1


class MyDNN:

    def __init__(self, architecture, Loss, weight_decay = 0):
        self.weight_decay = weight_decay
        self.Loss = Loss
        self.architecture = architecture
        self.params = {}
        self.actv = {}
        self.regularization = architecture[1]["regularization"]

        for idx, layer in enumerate(architecture):
            layer_num = idx + 1
            layer_input_dim = layer["input"]
            layer_output_dim = layer["output"]

            # init of weights and biases (uniform for w and zeros for biases)

            n = 1/np.sqrt(layer_input_dim)
            self.params['W' + str(layer_num)] = np.random.uniform(-n, n, (layer_output_dim, layer_input_dim))
            self.params['b' + str(layer_num)] = np.zeros((layer_input_dim, 1))
            self.actv['actv' + str(layer_num)] = layer["nonlinear"]


    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, learning_rate_decay=1.0,
                decay_rate=1, min_lr=0.0, x_val=None, y_val=None, ):
        # [x_norm, y_norm] = normalize(x_train, y_train)   # normalize
        for i in range(epochs):
            [x_shuf, y_shuf] = shuffle_data(x_train, ytrain)
            iterations = np.floor(x_train.shape[0]/batch_size)
            for k in range(iterations):
                x_batch = x_shuf[k*batch_size:((k+1)*batch_size-1),:]
                y_batch = y_shuf[k*batch_size:((k+1)*batch_size-1),:]
                y_tag = forwardprop(x_batch, y_batch, self.params, self.actv)
                [loss, accu] = calc_loss(y_batch, y_tag, self.Loss, self.params, self.regularization)
                backwardprop(self.params, loss[k])
            memory['weights_epoch'+str(i)] = self.params()
            loss_train[i] = loss
            accu_train[i] = accu
            y_tag_val = forwardprop(x_val, y_val, self.params, self.actv)
            [loss_val[i], accu_val[i]] = calc_loss(y_val, y_val_tag, self.Loss, self.params, self.regularization)
            

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
        activation_choosen = switcher.get(func,'invalid activation function')
        return activation_choosen(z)


activ = activations()

matansDNN = MyDNN(TempDNN, Loss, weight_decay)

print(matansDNN.actv)
print(matansDNN.params)


def shuffle_data(x, y)
    m = x.shape[1]
    xy = np.concatenate((x, y), axis=1)
    np.random.shuffle(xy)
    return xy[:,0:m-1], xy[:,m]


def frwdprop_layer(x, w, b, activation):
    z = np.dot(x, w)+b
    return activate(z, activation)


def forwardprop(x, y, params, activations):
    a_curr = x;
    for layer in length(1, params):
       a_new = frwdprop_layer(a_curr, params['w'+str(layer)], params['b'+str(layer)], actv['actv'+str(layer)])
       a_curr = a_new

