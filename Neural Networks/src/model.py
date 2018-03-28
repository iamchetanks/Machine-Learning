import numpy as np


def read_file(read):
    #read = "../data/downgesture_train.list"
    train = open(read,'r')
    train_data = []
    train_y = []
    for file_name in train:
        file_name = file_name.strip()
        file_name = "../data/"+file_name
        r = open(file_name,'r',encoding="latin-1")
        file_type = r.readline()
        created_by = r.readline()
        width, height = r.readline().strip().split()
        matrix = []
        for _ in range(int(height)):
            row = []
            for _ in range(int(width)):
                byte = r.read(1)
                if len(byte) != 0:
                    row.append(ord(byte))
            matrix.append(row)
        matrix = np.array(matrix)
        matrix_flatten = matrix.flatten()
        if matrix_flatten.shape[0] == 30:
            continue
        train_data.append(matrix_flatten)
        if 'down' in file_name:
            train_y.append(1)
        else:
            train_y.append(0)
    Y = np.array(train_y)
    Y = Y.reshape(Y.shape[0],-1).T
    train_data = np.vstack(train_data).T
    return train_data / 255, Y  # Standardize data to have feature values between 0 and 1.



def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1,
                  "W2": W2,
                  "b1": b1,
                  "b2": b2}
    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    #print("shape of Z", Z.shape)
    cache = (A, W, b)
    return Z, cache

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

def compute_cost(AL, Y):
    cost = np.sum((AL - Y)**2).squeeze()
    return cost

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> SIGMOID]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="sigmoid")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches



def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def main():
    X_train, Y_train = read_file("../data/downgesture_train.list")
    X_test, Y_test = read_file("../data/downgesture_test.list")
    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)
    print("X_test", X_test.shape)
    print("Y_test", Y_test.shape)

    n_x = X_train.shape[0]   # no of perceptrons in input layer
    n_h = 100                # no of perceptrons in hidden layer
    n_y = 1                  # output
    grads = {}
    learning_rate = 0.1
    no_of_iterations = 2000
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    m = X_train.shape[1]     # no of samples
    for i in range(no_of_iterations):
        A1, cache1 = linear_activation_forward(X_train, W1, b1, "sigmoid")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        #print(A2.shape)
        cost = compute_cost(A2, Y_train)

        # Initializing backward propagation
        dA2 = -(np.divide(Y_train, A2) - np.divide(1 - Y_train, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'sigmoid')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if i % 100 == 0:
            print("{}th iteration cost is {}".format(i,cost))

    print("Train data")
    pred_train = predict(X_train, Y_train, parameters)
    print("Accuracy: " + str(np.sum((pred_train == Y_train) / X_train.shape[1])))

    print("Test data")
    pred_test = predict(X_test, Y_test, parameters)
    print("Accuracy: " + str(np.sum((pred_test == Y_test) / X_test.shape[1])))


if __name__ == "__main__":
    main()