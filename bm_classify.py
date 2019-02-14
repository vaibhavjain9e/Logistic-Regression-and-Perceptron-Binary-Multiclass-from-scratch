import numpy as np
import copy

def indicator(arr):
    for i in range(len(arr)):
        if arr[i] <= 0:
            arr[i] = 1
        else:
            arr[i] = 0
    return arr

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """

    X = np.insert(X, X.shape[1], X.shape[0] * [1], axis=1)
    N, D = X.shape
    x = copy.deepcopy(y)

    for i in range(len(x)):
        if x[i] == 0:
            x[i] = -1

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    avg = float(step_size)/float(N)
    XnYn = X * x[:, None]
    negative_XnYn = -XnYn

    if loss == "perceptron":
        for iter in range(max_iterations):
            w = w + avg*np.sum((XnYn*indicator(np.dot(XnYn, w))[:, None]), axis=0)

    elif loss == "logistic":
        for iter in range(max_iterations):
            w = w + avg*np.sum(XnYn*sigmoid(np.dot(negative_XnYn, w))[:, None], axis=0)

    b = w[D-1]
    w = w[0:D-1]
    D = D-1
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """
    value = 1/(1 + np.exp(-z))

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    X = np.insert(X, D, N * [1], axis=1)
    w = np.append(w, b)
    result = np.zeros(N)
    
    if loss == "perceptron":
        result = np.dot(X, w)
        result[result > 0] = 1
        result[result <= 0] = 0

    elif loss == "logistic":
        result = sigmoid(np.dot(X, w))
        result[result >= 0.5] = 1
        result[result < 0.5] = 0

    return result


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    X = np.insert(X, X.shape[1], X.shape[0] * [1], axis=1)
    N, D = X.shape


    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    y_labels = list(np.unique(y))

    np.random.seed(42)

    if gd_type == "gd":
        w = np.zeros((C, D))
        b = np.zeros(C)
        y_prime = np.zeros((C, N))
        for i in range(N):
            index = y_labels.index(y[i])
            y_prime[index][i] = 1
        for iter in range(max_iterations):
            exp = np.exp(np.dot(w, np.transpose(X)))
            sum_value = np.sum(exp, axis=0)
            exp = exp/sum_value
            exp = exp - y_prime
            w = w - float(step_size/N)*np.dot(exp, X)

        b = w[:, -1]
        w = w[:, :-1]

    elif gd_type == "sgd":
        w = np.zeros((C, D))
        b = np.zeros(C)

        for iter in range(max_iterations):
            y_prime = np.zeros(C)
            idx = np.random.choice(N)
            idx_y = y_labels.index(y[idx])
            y_prime[idx_y] = 1

            intermediate = np.dot(w, X[idx])
            max_exp = np.max(intermediate)
            intermediate = intermediate - max_exp
            exp = np.exp(intermediate)
            sum_value = np.sum(exp)
            exp = exp / sum_value
            exp = exp - y_prime
            w = w - step_size*(exp[:, None])*(np.transpose((X[idx])[:, None]))

        b = w[:, -1]
        w = w[:, :-1]

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    X = np.insert(X, X.shape[1], X.shape[0] * [1], axis=1)
    N, D = X.shape
    w = np.insert(w, D - 1, b, axis=1)
    y = []
    C = np.dot(w, np.transpose(X))
    for i in range(C.shape[1]):
        y.append(np.argmax(C[:, i]))

    return np.array(y)




        