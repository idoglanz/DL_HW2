import numpy as np
import matplotlib as plt
import timeit

#------------------------------------------ DNN Class ------------------------------------------

class MyDNN:

    def __init__(self, architecture, Loss, weight_decay = 0):
        self.weight_decay = weight_decay
        self.Loss = Loss
        self.architecture = architecture
        self.params = {}
       # self.regularization = architecture[1]["regularization"]

        for idx, layer in enumerate(architecture):
            layer_num = idx + 1
            layer_input_dim = layer["input"]
            layer_output_dim = layer["output"]

            # init of weights and biases (uniform for w and zeros for biases)

            n = 1/np.sqrt(layer_input_dim)
            self.params['W' + str(layer_num)] = np.random.uniform(-n, n, (layer_output_dim, layer_input_dim))
            self.params['b' + str(layer_num)] = np.zeros((layer_input_dim, 1))
            self.params['actv' + str(layer_num)] = layer["nonlinear"]
            self.params['regularization' + str(layer_num)] = layer['regularization']

    def fit(self, x_train, y_train, epochs, batch_size, learning_rate, learning_rate_decay=1.0,
                decay_rate=1, min_lr=0.0, x_val=None, y_val=None, ):
        # [x_norm, y_norm] = normalize(x_train, y_train)   # normalize
        history = {}
        for i in range(epochs):
            [x_shuf, y_shuf] = shuffle_data(x_train, ytrain)
            iterations = np.floor(x_train.shape[0]/batch_size)

            start = timeit.default_timer()

            # Run training for epoch:

            for k in range(iterations):
                x_batch = x_shuf[k*batch_size:((k+1)*batch_size-1),:]
                y_batch = y_shuf[k*batch_size:((k+1)*batch_size-1),:]

                [y_tag, history] = forwardprop(x_batch, self.params)

                [loss, accu] = calc_loss(y_batch, y_tag, self.Loss, self.params, self.regularization)

                gradients = backwardprop(self.params, loss[k], y_batch, y_tag, history)

                lr = max(learning_rate*learning_rate_decay**(k/decay_rate), min_lr)
                update_weights(self.params, gradients, lr)

            stop = timeit.default_timer()

            # Save all relevant info from epoch:

            if x_val is not None:
                [y_tag_val, history_val] = forwardprop(x_val, self.params)
                [loss_val[i], accu_val[i]] = calc_loss(y_val, y_tag_val, self.Loss, self.params, self.regularization)
                history['val_loss' + str(i)] = loss_val[i]
                history['val_accu' + str(i)] = accu_val[i]

            history['weights_epoch' + str(i)] = self.params()
            history['loss' + str(i)] = loss
            history['accu' + str(i)] = accu
            history['runtime' + str(i)] = stop-start

            # Print status message after every epoch

            print(['Epoch ' + str(i) + '/' + str(epochs) + ' - ' + str(stop-start) + ' Seconds - loss: '
                   + str(loss_train[i]) + ' - acc: ' + str(accu_train[i]) + ' - val_loss: ' + str(loss) +
                   ' - val_acc: ' + str(accu)])

        return self.params


def predict(self,X, batch_size=None):

    if batch_size is not None:
        batches = X.size[0] / batch_size
    else:
        batches = 1

    for k in range(batches):
        pred[k] = forwardprop(x, self.params)

    return pred


def evaulate(self, x, y, batch_size = None):

    y_bar = predict(X, batch_size)
    [loss[k], accu[k]] = calc_loss(y, y_bar, self.Loss, self.params)


    return loss, accu

#-------------------------------------- Activation class -----------------------------------------

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

    def relu(self, z, back=False):
        if back is False:
            return np.maximum(z, 0)
        else:
            new_z = z[z<=0] = 0
            return new_z

    def softmax(self, z, back=False):
        return np.exp(z)/np.sum(np.exp(z))

    def none(self,z, back=False):
        if back is False:
            return z
        else:
            return 1


#------------------------------------------ functions ------------------------------------------


def activate(z, func, back = False):
        switcher = {
                  'relu': activ.relu,
                  'sigmoid': activ.sigmoid,
                  'softmax': activ.softmax,
                  'none': activ.none
        }
        activation_choosen = switcher.get(func,'invalid activation function')
        if back is False:
            return activation_choosen(z)
        else:
            return activation_choosen(z, True)


def shuffle_data(x, y):
    m = x.shape[1]
    xy = np.concatenate((x, y), axis=1)
    np.random.shuffle(xy)
    return xy[:,0:m-1], xy[:,m]


def frwdprop_layer(x, w, b, activation):
    z = np.dot(x, w)+b

    return activate(z, activation), z


def forwardprop(x, params):
    a_curr = x
    history = {}

    for layer in length(1, params):
       [a_new, z_new] = frwdprop_layer(a_curr, params['w'+str(layer)], params['b'+str(layer)], params['actv'+str(layer)])
       a_curr = a_new
       history["A" + str(layer)] = a_new
       history["z" + str(layer-1)] = z_new

    return a_curr, history


def backprop_layer(dA_curr, W, z_curr, A_prev, activation):
    m = A_prev.shape[1]

    # to update W we need: dL/dA_curr(=dA_curr) * dA_curr/dsigma(=dA_curr) * dsigma/dz(=activation') * dz/dw(=A_prev)
    # to update b we need: dL/dA_curr(=dA_curr) * dA_curr/dsigma(=dA_curr) * dsigma/dz(=activation') * dz/dw(=A_prev)

    dL_dz = dA_curr * activate(z_curr, activation, back)
    dL_dw = np.dot(dL_dz, A_prev.T)/m
    dL_db = np.sum(dL_dz, axis=1, keepdims=True) / m
    dL_dA_prev = np.dot(W.T, dsigma_dz)    # will be needed for previous layer

    return dL_dA_prev, dL_dw, dL_db

def backprop(params, loss, y_batch, y_tag, history):
    gradients = {}
    m = y_batch.shape[1]
    y = y_batch.reshape(Y_tag.shape)
    if loss == 'CrossEntropy':
        dL_dy = - (np.divide(y_batch, y_tag) - np.divide(1 - y_batch, 1 - y_tag))  # NEEDS TO BE VARIFIED!!

    elif loss == 'MSE':
        dL_dy = - (1/m)*np.sum(np.dot(y-y_tag,(y-y_tag).T))

    else:
        raise Exception('Loss function not supported')

    dA_prev = dL_dy  # gradient of previous state (in last layer = y)

    for prev_idx, layer in reversed(list(enumerate(params))):
        curr_layer = prev_idx + 1
        dA_curr = dA_prev

        A_prev = history["A" + str(layer_idx_prev)]
        z_curr = history["Z" + str(layer_idx_curr)]

        [dA_prev, dw, db] = backprop_layer(dA_curr, params['W'+str(curr_layer)], z_curr, A_prev, params['actv' + str(curr_layer)])

        gradients['dW' + str(curr_layer)] = dw
        gradients['db' + str(curr_layer)] = db

    return gradiants


def update_weights(params, gradients, lr):

    for index, layer in enumerate(params):
        params["W" + str(index)] -= lr * gradients["dW" + str(index)]
        params["b" + str(index)] -= lr * gradients["db" + str(index)]

    return params


def calc_loss(y, y_bar, loss_func, params):

    regularization_value = weights_weight(params)
    m = size(y)[0]

    if loss_func is 'MSE':    # for regression
        accu = None
        loss = -(1/m)*np.dot((y-ybar), (y-y_bar).T) + regularization_value

    elif loss_func is 'cross_entropy':  # for classification
        loss = 1 / m * (np.dot(y, np.log(y_bar).T) + np.dot(1 - y, np.log(1 - y_bar).T)) + regularization_value
        prob = np.copy(y_bar)
        prob[prob > 0.5] = 1
        prob[prob <= 0.5] = 0
        accu = (y_bar_ == y).all(axis=0).mean()

    else:
        loss = None
        accu = None

    return loss, accu


def weights_weight(params):
    w_total = np.zeros(param['W1'],1)

    for idx, layer in params:
        if params['regularization' + str(idx)] is 'l1':
            w_total += np.prod(params['W' + str(idx)], axis=1)
        elif params['regularization' + str(idx)] is 'l2':
            w_total += np.prod(np.power(params['W' + str(idx)], 2), axis=1)
        elif params['regularization' + str(idx)] is 'None':
            w_total = w_total

    return np.mean(w_total)


############################################################################################################

def find_mean(data_set):
    train_mean = np.mean(data_set, axis=0)
    return train_mean

############################################################################################################

def preprocess(data_set, train_mean):
    data_set_processed = data_set - train_mean
    return data_set_processed

############################################################################################################

def class_process(y):
    y = [(1 if (y[i]%2 == 0) else -1) for i in range(len(y))]
    y = np.array(y)
    return y



#------------------------------------------ Main ------------------------------------------

# Load dataset

data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
urllib_request.urlretrieve(data_url, "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')




TempDNN = [{"input": 2, "output": 4, "nonlinear": "relu", "regularization": "l2"},
           {"input": 4, "output": 5, "nonlinear": "sigmoid", "regularization": "l2"},
           {"input": 5, "output": 1, "nonlinear": "softmax", "regularization": "l1"}]

Loss = 'MSE'
weight_decay = 0.1

activ = Activations()

DNN = MyDNN(TempDNN, Loss, weight_decay)


#print(DNN.actv)
print(DNN.params['W' + str(2)])
