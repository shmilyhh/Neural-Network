import numpy as np
import matplotlib.pyplot as pyplot
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

np.random.seed(1)

"""Initialization
"""


def initialize_parameters_deep(layer_dims):
    """Initialize the parameters.

    Arguments:
        layer_dims {list} -- a list of dimensions of layers.

    Returns:
        parameters {dict} -- {"W1", "b1", ..., "Wn", "bn"}
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" +
                   str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters["W" + str(l)].shape ==
               (layer_dims[l], layer_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters


"""Forward Propagation Module
1. LINEAR
2. LINEAR -> ACTIVATION
3. [LINEAR -> RELU] X (L-1) -> LINEAR -> SIGMOID 
"""
# Linear Forward


def linear_forward(A, W, b):
    """Implement the linear part of a layer's forward propagation.

    Arguments:
        A {np.array} -- activations from previous layer
        W {np.array} -- weight of this layer
        b {np.array} -- bias of this layer

    Returns:
        Z {np.array} -- the input of the activation function
        cache {tuple} -- ("A", "W", "b")
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

# Linear Activation Forward


def linear_activation_forward(A_prev, W, b, activation):
    """Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
        A_prev {np.array} -- activations from previous layer
        W {np.array} -- weight matrix
        b {np.array} -- bias vector
        activation {str} -- the activation name

    Returns:
        A {np.array} -- the output of the activation function
        cache {tuple} -- ("linear_cache", "activation_cache")
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    if activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))

    cache = (linear_cache, activation_cache)
    return A, cache

# L Model Forward


def L_model_forward(X, parameters):
    """Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    Arguments:
        X {np.array} -- input
        parameters {dict} -- initialization of the parameters

    Returns:
        AL {np.array} -- last post-activation value
        caches {list} -- list of caches containing every cache of linear_activation_forward()
    """

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "sigmoid")
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


"""Cost Function
"""


def compute_cost(AL, Y):
    """Implement the cost function

    Arguments:
        AL {np.array} -- the last output of the model
        Y {np.array} -- true label vector

    Returns:
        cost {float} -- cross-entropy cost
    """

    m = Y.shape[1]

    cost = -1 * (1 / m) * np.sum(np.multiply(Y, np.log(AL)) +
                                 np.multiply((1 - Y), np.log((1 - AL))),
                                 axis=1, keepdims=True)

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost


"""Backward Propagation Module
1. LINEAR backward
2. LINEAR -> ACTIVATION
3. [LINEAR -> RELU] X (L-1) -> LINEAR -> SIGMOID
"""
# Linear Forward


def linear_backward(dZ, cache):
    """Implement the linear portion of backward propagation for a single layer

    Arguments:
        dZ {np.arrray} -- gradient of cost with respect to the linear output
        cache {tuple} -- (A_prev, W, b) from the forward propagation

    Returns:
        dA_prev {np.array} -- gradient of the cost with respect to the activation
        dW {np.array} -- gradient of the cost with respect to the weight
        db {np.array} -- gradient of the cost with respect to the bias
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, np.transpose(A_prev))
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(np.transpose(W), dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

# Linear Activation Backward


def linear_activation_backward(dA, cache, activation):
    """Implement the backward propagation for the LINEAR->ACTIVATION layer

    Arguments:
        dA {np.array} -- post-activation gradient for current layer l
        cache {tuple} -- tuple of values
        activation {str} -- activation name

    Returns:
        dA_prev {np.array} -- gradient of the cost with respect to the activation
        dW {np.array} -- gradient of the cost with respect to the weight
        db {np.array} -- gradient of the cost with respect to the bias
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# L Model Backward


def L_model_backward(AL, Y, caches):
    """Implement the backward propagation for [LINEAR->RELU] * (L-1) -> LINEAR->sigmoid

    Arguments:
        AL {np.array} -- the last output of the model
        Y {np.array} -- true label
        caches {list} -- list of caches containing [("linear cache", "activation cache"), ...]

    Returns:
        grads {dict} -- dictionary with the gradients {"dA", "db", "dA"}
    """

    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    # initialize the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. dAL, current_cache -> dA(L-1), dW, db
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)
                                                        ] = linear_activation_backward(dAL, current_cache, "sigmoid")

    # relu loop from l-2 to 0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients. dA(l+1), current_cache -> dA(l), dW(l+1), db(l+1)
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dw" + str(l+1)], grads["db" + str(l+1)
                                                            ] = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")

    return grads


"""Update Parameters
"""


def update_parameters(parameters, grads, linearing_rate):
    """Update parameters

    Arguments:
        parameters {dict} -- {"W", "b"}
        grads {dict} -- {"dW", "db", "dA"}
        linearing_rate {float} -- learning rate

    Return:
        parameters {dict} -- {"W", "b", "A"}            
    """

    L = len(parameters)

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
            linearing_rate * parameters["W" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
            linearing_rate * parameters["b" + str(l+1)]
    return parameters
