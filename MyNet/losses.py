import numpy as np


# loss function and its derivative
def mse(y_pred, y_true):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


def cross_entropy(y_pred, y_true):
    """
    cross entropy loss
    :return: -np.log(output[label])
    """
    max_p = np.argmax(y_true)
    return -np.log(y_pred[max_p])


def cross_entropy_prime(y_pred, y_true):
    """
    derivative of cross entropy loss
    :return: gradient[label] = -1 / output[label]
    """
    max_p = np.argmax(y_true)
    y = np.zeros(len(y_true))
    y[max_p] = -1 / y_pred[max_p]

    return y


def cross_entropy_with_softmax_prime(activated_out, y_true):
    """
    derivative of cross entropy loss
    :return: gradient[label] = -1 / output[label]
    """
    max_p = np.argmax(y_true)
    y = np.zeros(len(y_true))
    y[max_p] = 1

    return activated_out - y
