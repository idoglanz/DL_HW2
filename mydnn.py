import numpy as np
import matplotlib.pyplot as plt
import timeit
import pickle, gzip, json, urllib.request

# ------------------------------------------ DNN Class ------------------------------------------


class MyDNN:

    def __init__(self, architecture, Loss, weight_decay=0.0):
        self.weight_decay = weight_decay
        self.Loss = Loss
        self.architecture = architecture
        self.params = {}

        for idx, layer in enumerate(architecture):
            layer_num = idx + 1
            layer_input_dim = layer["input"]
            layer_output_dim = layer["output"]

            # init of weights and biases (uniform for w and zeros for biases)
            n = 1/np.sqrt(layer_input_dim)
            self.params['W' + str(layer_num)] = np.random.uniform(-n, n, (layer_output_dim, layer_input_dim))
            self.params['b' + str(layer_num)] = np.zeros((layer_output_dim, 1))
            self.params['actv' + str(layer_num)] = layer["nonlinear"]
            self.params['regularization' + str(layer_num)] = layer['regularization']

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, learning_rate_decay=1.0,
            decay_rate=1, min_lr=0.0, x_val=None, y_val=None, ):

        # history = {}

        loss_vec = []
        loss_vec_val = []
        accu_vec = []
        accu_vec_val = []

        for i in range(epochs):

            # shuffle data on every epoch (shuffling x and y together)
            # [x_shuf, y_shuf] = shuffle_data(x_train, y_train)
            [x_shuf, y_shuf] = [x_train, y_train]

            # calc number of iteration (batch size dependent)
            iterations = np.floor(x_train.shape[0]/batch_size)

            start = timeit.default_timer()

            # Run training for epoch:

            for k in range(int(iterations)):

                x_batch = x_shuf[k*batch_size:((k+1)*batch_size), :]
                y_batch = y_shuf[k*batch_size:((k+1)*batch_size), :]

                [y_bar_batch, history] = forwardprop(x_batch, self.params, self.architecture)

                [loss, accu] = calc_loss(y_batch, y_bar_batch, self.Loss, self.params, self.architecture, self.weight_decay)

                loss_vec.append(float(loss))
                accu_vec.append(accu)

                if x_val is not None:

                    [y_tag_val, history_val] = forwardprop(x_val, self.params, self.architecture)
                    [loss_val, accu_val] = calc_loss(y_val, y_tag_val, self.Loss, self.params, self.architecture,
                                                     self.weight_decay)
                    loss_vec_val.append(float(loss_val))
                    accu_vec_val.append(accu_val)

                gradients = backprop(self.params, self.Loss, y_batch, y_bar_batch, history, self.architecture, self.weight_decay)

                lr = max(learning_rate*learning_rate_decay**(k/decay_rate), min_lr)

                self.params = update_weights(self.params, gradients, lr, self.architecture)

            stop = timeit.default_timer()

            #  test validation set and save all relevant info from epoch:
            # print_output(np.argmax(y_batch, axis=1), np.argmax(y_tag, axis=1), i)

            if x_val is not None:
                [y_tag_val, history_val] = forwardprop(x_val, self.params, self.architecture)
                [loss_val, accu_val] = calc_loss(y_val, y_tag_val, self.Loss, self.params, self.architecture, self.weight_decay)
                history['val_loss' + str(i)] = loss_val
                history['val_accu' + str(i)] = accu_val
                # print_output(np.argmax(y_val, axis=1), np.argmax(y_tag_val, axis=1), i)

            history['weights_epoch' + str(i)] = self.params
            history['loss' + str(i)] = loss
            history['accu' + str(i)] = accu
            history['runtime' + str(i)] = stop-start

            # Print status message after every epoch

            print(['Epoch ' + str(i+1) + '/' + str(epochs) + ' - ' + str(stop-start) + ' Seconds - loss: '
                   + str(loss) + ' - acc: ' + str(accu) + ' - val_loss: ' + str(loss_val) +
                   ' - val_acc: ' + str(accu_val)])

        # save loss and accuracies vs iterations

        history['losses'] = loss_vec
        history['accus'] = accu_vec
        history['losses_val'] = loss_vec_val
        history['accus_val'] = accu_vec_val

        return self.params, history

    def predict(self, x, y, batch_size=None):

        pred_shape = y.shape

        if batch_size is not None:

            batch_num = np.floor(pred_shape[0] / batch_size)

            y_bar = np.zeros((pred_shape[0], pred_shape[1]))

            for iter in range(batch_num):

                x_batch = x[iter * batch_size:(iter+1) * batch_size, :]

                [y_bar[iter * batch_size:(iter+1) * batch_size, :], history] = forwardprop(x_batch, self.params, self.architecture)

        else:
            # y_bar = np.zeros((pred_shape[0], pred_shape[1]))
            [y_bar, history] = forwardprop(x, self.params, self.architecture)

        return y_bar

    def evaluate(self, x, y, batch_size=None):

        y_bar = self.predict(x, y, batch_size)
        [loss, accu] = calc_loss(y, y_bar, self.Loss, self.params, self.architecture, weight_decay=0)
        return loss, accu, y_bar


# -------------------------------------- Activation class -----------------------------------------

class Activations:

    def __init__(self):
        self.value = 0
        self.back = None

    def sigmoid(self, z, back=False):
        sig = 1 / (1 + np.exp(-z))
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

    def softmax(self, z, back=False):   # stable version

        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        temp = exps / np.sum(exps, axis=1, keepdims=True)

        if back is False:
            return temp
        else:
            z_grad = np.zeros((z.shape[1], z.shape[1]))
            for k in range(z.shape[0]):
                s = temp[k, :].reshape(-1, 1)
                z_grad += np.diagflat(s) - np.dot(s, s.T)
            return (1/z.shape[0])*z_grad
            # return temp*(np.eye(z.shape[0], z.shape[1])-temp)

    def none(self, z, back=False):
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


def shuffle_data(x, y):
    m = x.shape[1]
    xy = np.concatenate((x, y), axis=1)
    np.random.shuffle(xy)
    x_new = xy[:, :m]
    y_new = xy[:, m:]
    return x_new, y_new


def frwdprop_layer(x, w, b, activation):
    z = np.dot(x, w.T) + b.T
    return activate(z, activation), z


def forwardprop(x, params, architecture):
    a_curr = x
    history_local = {}

    for idx, layer in enumerate(architecture):
        [a_new, z_new] = frwdprop_layer(a_curr, params['W'+str(idx+1)], params['b'+str(idx+1)], params['actv'+str(idx+1)])

        history_local["A" + str(idx)] = a_curr
        history_local["z" + str(idx+1)] = z_new

        a_curr = a_new

    return a_curr, history_local


def backprop_layer(dA_curr, W, z_curr, A_prev, activation, last=False):

    m = dA_curr.shape[0]

    # to update W we need: dL/dA_curr(=dA_curr) * dA_curr/dsigma(=dA_curr) * dsigma/dz(=activation') * dz/dw(=A_prev)
    # to update b we need: dL/dA_curr(=dA_curr) * dA_curr/dsigma(=dA_curr) * dsigma/dz(=activation') * dz/dw(=A_prev)

    if activation is 'softmax':
        if last is True:   # in the case last layer is softmax (hence it is followed by cross entropy)
            dL_dz = dA_curr
        else:              # cases where softmax is used not on last layer
            dL_dz = np.dot(dA_curr, activate(z_curr, activation, True))
    else:
        dL_dz = dA_curr * activate(z_curr, activation, True)

    dL_dw = (1/m)*np.dot(dL_dz.T, A_prev)
    dL_db = np.sum(dL_dz, axis=0, keepdims=True) / m
    dL_dA_prev = np.dot(dL_dz, W)    # will be needed for previous layer

    return dL_dA_prev, dL_dw, dL_db.T


def backprop(params, loss, y_batch, y_tag, history, architecture, weight_decay):
    gradients = {}
    m = y_batch.shape[0]

    if loss == 'cross_entropy':  # also combining the softmax grad (reducing calc time)

          dL_dy = y_tag
          dL_dy[range(m), y_batch.argmax(axis=1)] -= 1

    elif loss == 'MSE':
        dL_dy = - np.sum(y_batch - y_tag, axis=1)
        dL_dy = convert_vector_to_matrix(dL_dy)

    else:
        raise Exception('Loss function not supported')

    dA_prev = dL_dy  # gradient of previous state (in last layer = y)
    last_flag = True
    for prev_idx, layer in reversed(list(enumerate(architecture))):
        curr_layer = prev_idx + 1
        dA_curr = dA_prev

        A_prev = history["A" + str(prev_idx)]
        z_curr = history["z" + str(curr_layer)]

        [dA_prev, dw, db] = backprop_layer(dA_curr, params['W'+str(curr_layer)], z_curr, A_prev, params['actv' + str(curr_layer)], last_flag)

        gradients['dW' + str(curr_layer)] = dw + (weight_decay/m) * layer_weight(params['W'+str(curr_layer)], params['regularization' + str(curr_layer)])

        gradients['db' + str(curr_layer)] = db
        last_flag = False

    return gradients


def update_weights(params, gradients, lr, architecture):
    for index, layer in enumerate(architecture, 1):
        params['W' + str(index)] -= lr * gradients["dW" + str(index)]
        params['b' + str(index)] -= lr * gradients["db" + str(index)]

    return params


def calc_loss(y, y_bar, loss_func, params, architecture, weight_decay):

    m = y.shape[0]
    regularization_value = (weight_decay/m) * weights_weight(params, architecture)

    if loss_func is 'MSE':    # for regression
        accu = None
        loss = -(1 / m) * np.dot((y-y_bar).T, (y-y_bar)) + regularization_value

    elif loss_func is 'cross_entropy':  # for classification
        loss = -(1 / m) * np.trace(np.dot(y, np.log(y_bar).T)) + regularization_value

        guess = np.zeros((y_bar.shape[0], 1))
        guess[np.argmax(y_bar, axis=1) == np.argmax(y, axis=1)] = 1
        accu = np.sum(guess)/m

    else:
        loss = None
        accu = None

    return loss, accu


def layer_weight(w, regularization):
    if regularization is 'l1':
        return np.sign(w)
    elif regularization is 'l2':
        return 2*w
    else:
        return w


def weights_weight(params, architecture):

    w_total = 0

    for idx, layer in enumerate(architecture, 1):
        if params['regularization' + str(idx)] is 'l1':
            w_total += np.sum(np.absolute(params['W' + str(idx)]))

        elif params['regularization' + str(idx)] is 'l2':
            w_total += np.sum(np.power(params['W' + str(idx)], 2))

        elif params['regularization' + str(idx)] is 'None':
            w_total = w_total

    return w_total


def print_output(y, ybar, epoch, blck=False):
    plt.clf()
    x_axis = np.linspace(1, y.shape[0], y.shape[0])
    window = [1, 500]

    plt.plot(x_axis[window[0]:window[1]], y[window[0]:window[1]], 'ro')
    plt.plot(x_axis[window[0]:window[1]], ybar[window[0]:window[1]], 'b+')
    plt.ylabel("Class")
    plt.xlabel("Sample")
    if blck is not False:
        plt.title(["y vs y_bar, Test Set"])
    else:
        plt.title(["y vs y_bar, epoch: " + str(epoch)])
    plt.legend(["Y true", "Y bar"])
    plt.ion()
    plt.pause(0.1)
    plt.show(block=blck)


def print_result(loss, accu, loss_val, accu_val, batch_size):

    x_axis = np.linspace(1, len(loss), len(loss))

    ls = plt.figure(2)
    plt.plot(x_axis, loss, 'r')
    plt.plot(x_axis, loss_val, 'b')
    plt.legend(["Train set", "Validation set"])
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title(["Loss VS iterations - batch size = " + str(batch_size)])
    ls.show()

    if accu_val is not None:
        ac = plt.figure(3)
        plt.plot(x_axis, accu, 'r')
        plt.plot(x_axis, accu_val, 'b')
        plt.legend(["Train set", "Validation set"])
        plt.ylabel("Accuracy")
        plt.xlabel("Iterations")
        plt.title(["Accuracy VS iterations - batch size = " + str(batch_size)])
        ac.show()

    plt.ion()
    plt.pause(0.1)
    plt.show(block=True)

############################################################################################################


def find_mean(data_set):
    train_mean = np.mean(data_set, axis=0)
    return train_mean

############################################################################################################


def preprocess(data_set, train_mean):
    data_set_processed = data_set - train_mean
    return data_set_processed

############################################################################################################

def class_process_mnist(y, classes):

    y_new = np.zeros((len(y), classes))
    row = np.linspace(0, y.shape[0]-1, y.shape[0], dtype='int')
    col = y
    y_new[row, col] = 1

    return y_new

############################################################################################################


def convert_vector_to_matrix(y):
    y = y[:, np.newaxis]
    return y


activ = Activations()

# ------------------------------------------ Main ------------------------------------------

# # Load data-set and preprocess (normalize)
#
# # data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
# # urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
# with gzip.open('mnist.pkl.gz', 'rb') as f:
#     train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
#
# train_mean = find_mean(train_set[0])
#
# X_ = preprocess(train_set[0], train_mean)
# X_val = preprocess(valid_set[0], train_mean)
# X_test = preprocess(test_set[0], train_mean)
#
# y_ = class_process_mnist(train_set[1], 10)
# y_val = class_process_mnist(valid_set[1], 10)
# y_test = class_process_mnist(test_set[1], 10)
#
# # construct network architecture
#
# TempDNN = [{"input": 784, "output": 800, "nonlinear": "relu", "regularization": "l2"},
#            {"input": 800, "output": 300, "nonlinear": "relu", "regularization": "l2"},
#            {"input": 300, "output": 10, "nonlinear": "softmax", "regularization": "l2"}]
#
# Loss = 'cross_entropy'
# weight_decay = 0.001
#
# # init DNN and activations
#
# activ = Activations()
#
# DNN = MyDNN(TempDNN, Loss, weight_decay)
#
#
# [trained_params, history1] = DNN.fit(X_, y_, epochs=5, batch_size=1000, learning_rate=0.1, learning_rate_decay=1, decay_rate=1, min_lr=0.0, x_val=X_val, y_val=y_val)
#
# [loss, accu, y_bar] = DNN.evaluate(X_test, y_test, None)
#
# print(loss, accu)
#
# print_output(np.argmax(y_test, axis=1), np.argmax(y_bar, axis=1), 0, blck=True)

