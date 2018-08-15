import numpy as np

# get layer sizes


def layer_size(X, Y):
    """Get layer size.

    Arguments:
        X {np.array} -- input
        Y {np.array} -- label

    Returns:
        n_x {int} -- size of input X
        n_h {int} -- size of hidden layer
        n_y {int} -- size of output Y
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

# initialize parameters


def initialize_parameters(n_x, n_h, n_y):
    """Initialize the parameters.

    Arguments:
        n_x {int} -- size of input X
        n_h {int} -- size of hidden layer
        n_y {int} -- size of output Y

    Returns:
        parameters {dict} -- {'W1', 'b1', 'W2', 'b2'}
    """

    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    """Forward propagation.

    Arguments:
        X {np.array} -- input
        parameters {dict} -- {"W1", "b1", "W2", "b2"}

    Returns:
        A2 {np.array} -- output
        cache {dict} -- {"Z1", "A1", "Z2", "A2"}
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmod(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache


def compute_cost(A2, Y, parameters):
    """Compute cost.

    Arguments:
        A2 {np.array} -- output
        Y {np.array} -- label
        parameters {dict} -- {"W1", "b1", "W2", "b2"}

    Returns:
        cost {float} -- cross-entropy cost given equation
    """

    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y)
    cost = - np.sum(logprobs)

    return cost


def backward_propagation(parameters, cache, X, Y):
    """Backward propagation.

    Arguments:
        parameters {dict} -- {"W1", "b1", "W2", "b2"}
        cache {dict} -- {"Z1", "A1", "Z2", "A2"}
        X {np.array} -- input
        Y {np.array} -- output

    Returns:
        grads {dict} -- {"dZ1", "db1", "dZ2", "db2"}
    """
    m = Y.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, np.transpose(A1))
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(np.transpose(W2), dZ2), (1 - np.power(A1, 2)))
    dW1 = 1/m * np.dot(dZ1, np.transpose(X))
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """Update parameters.

    Arguments:
        parameters {dict} -- {"W2", "b2", "W1", "b1"}
        grads {dict} -- {"dW2", "db2", "dW1", "db1"}

    Keyword Arguments:
        learning_rate {float} -- learning rate (default: {1.2})

    Returns:
        parameters {dict} -- {"W2", "b2", "W1", "b1"}
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def nn_model(X, Y, num_iterations=10000, print_cost=True):
    """model.

    Arguments:
        X {np.array} -- input
        Y {np.array} -- output

    Keyword Arguments:
        num_iterations {int} -- iterations (default: {10000})

    Returns:
        parameters {dict} -- {"W1", "b1", "W2", "b2"}
    """
    np.random.seed(3)
    n_x = layer_size(X, Y)[0]
    n_y = layer_size(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f", cost)
    return parameters


def predict(parameters, X):
    """Use the learned parameters, predicts a class for each example in X.

    Arguments:
        parameters {dict} -- {"W1", "b1", "W2", "b2"}
        X {np.array} -- input

    Returns:
        predictions {np.array} -- vector of predictions of our model (0/1)
    """

    A2, cache = forward_propagation(parameters, X)
    predictions = (A > 0.5)

    return predictions
