import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(a):
    """
    sigmoid activation function
    :return: 1 / (1 + np.exp(-a))
    """
    return 1 / (1 + np.exp(-a))


def sigmoid_prime(a):
    """
    derivative of sigmoid function
    :return: sigmoid(a) * (1 - sigmoid(a))
    """
    return sigmoid(a) * (1 - sigmoid(a))


def softmax(a):
    """
    softmax activation function
    :return: np.exp(a) / np.sum(np.exp(a), axis=0)
    """
    return np.exp(a) / np.sum(np.exp(a))


def softmax_prime(a):
    """
    softmax activation function
    :return: np.exp(a) / np.sum(np.exp(a), axis=0)
    """
    return a


def softmax_dummy_prime(a):
    """
    softmax activation function
    :return: np.exp(a) / np.sum(np.exp(a), axis=0)
    """
    return a


# TODO: add relu

def relu(a):
    return np.maximum(0, a)


def relu_prime(a):
    return (a > 0).astype(a.dtype)
