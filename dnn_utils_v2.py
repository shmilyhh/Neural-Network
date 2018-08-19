import numpy as np

def sigmoid(Z):
    """Implements the sigmoid activation in numpy

    Arguments:
        Z {np.array} -- input

    Returns:
        A {np.array} -- output
        cache {np.array} -- cache Z for backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """Implement the RELU function
    
    Arguments:
        Z {np.array} -- input
    
    Returns:
        A {np.array} -- output
        cache {np.array} -- cache Z for backpropagation
    """

    A = np.maximum(0, Z)

    cache = Z

    return A, cache

def sigmoid_backward(dA, cache):
    """Implement the backward propagation for a single Sigmoid Unit
    
    Arguments:
        dA {np.array} -- post-activation gradient
        cache {np.array} -- Z
    
    Returns:
        dZ {np.array} -- gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 - np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def relu_backward(dA, cache):
    """Implement the backward propagation for a single Relu Unit
    
    Arguments:
        dA {np.array} -- post-activation gradient
        cache {np.array} -- Z
    
    Returns:
        dZ {np.array} -- gradient of the cost with respect to Z
    """

    Z = cache

    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ