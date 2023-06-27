import random

import numpy as np
from matplotlib import pyplot as plt


class Net:
    """
    the network model.
    holds all layers.
    forward and backward propagate layers.
    takes data to train on.
    can predict outcomes.
    """

    def __init__(self, layers, loss="cross_entropy"):
        """
        initialize Network
        set layers and loss function
        :param layers: tuple of Layers used in the net
        :param loss: loss function to use
        """

        losses = {
            "cross_entropy": (cross_entropy, derivative_cross_entropy),
            "mse": (mse, derivative_mse)
        }

        # set loss function and its derivative
        self.loss_function, self.derivative_loss_function = losses.get(loss)

        # set layers
        self.layers = layers

    def forward(self, image):
        """
        forward propagate all layers
        :return: output
        """

        # flatten 2D input to 1D
        output = image.flatten()

        # propagate through all layers
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, dL_dy, **kwargs):
        """
        backwards propagate and update parameters for all layers
        :param dL_dy: gradient of Loss function with respect to the output
        of the network
        """

        # gradient of Loss function with respect to weights, can be computed
        # by using the chain rule

        # first term of this gradient is the gradient of Loss with respect to
        # the output of the last layer.
        grad = dL_dy

        # we than propagate backwards over all layers, beginning with last

        # in the backward method of Layer the weights and biases get and
        # the gradient of Loss with respect to the previous layer gets
        # returned
        # this gradient feeds into the backward method of the next layer
        # and so on

        for layer in self.layers[::-1]:
            grad = layer.backward(grad, **kwargs)

    def train(self, image, label, alpha=0.001):
        """
        train the network with one set of image and label
        :param alpha: learning rate
        :return: loss for given training sample
        """

        # feed forward to get output and loss

        output = self.forward(image)

        # compute loss
        loss = self.loss_function(output, label)

        # compute gradient of loss function
        grad = self.derivative_loss_function(output, label)

        # backwards propagate net and update weights and biases
        self.backward(grad, alpha=alpha)

        return loss

    def fit(self, data, epochs=20):
        """
        trains model with data
        :param epochs number of epochs = how many times the network gets
        trained with the whole data

        :return: train_losses, eval_losses
        """

        # number of epochs

        # initialize arrays for losses
        train_losses = np.zeros(epochs)
        eval_losses = np.zeros(epochs)

        # split data into training and validation data
        train_data, eval_data = split_data(data)

        print("Fitting model to data.")
        print("Number of epochs:", epochs)

        # train net for each epoch
        for j in range(epochs):
            print("epoch:", j + 1)

            # shuffle data for each new epoch
            random.shuffle(train_data)

            # train and store losses
            for label, image in train_data:
                train_losses[j] += self.train(image, label) / len(
                    train_data) / epochs

            # validate
            for label, image in eval_data:
                # compute and store losses
                output = self.forward(image)
                eval_losses[j] += self.loss_function(output, label) / len(
                    eval_data) / epochs

            # plot losses after each epoch to keep track of progression
            plt.plot(range(len(train_losses)), train_losses,
                     label="train_losses")
            plt.plot(range(len(eval_losses)), eval_losses,
                     label="eval_losses")

            plt.legend()
            plt.show()

        print("Finished Training.")

        return train_losses, eval_losses

    def predict(self, image):
        """
        predict label for given image
        :return: predicted label, output
        """

        out = self.forward(image)

        # the predicted label is the highest value of the output array
        pred = np.argmax(out)

        return pred, out


class Layer:
    """
    Base class for Layers.
    Standard Dense Layer.
    activation = sigmoid
    """

    def __init__(self, input_units, output_units):
        """
        initialize layers with random weights and biases
        """

        self.weight = np.random.randn(input_units, output_units) / input_units
        self.bias = np.random.randn(output_units)

        # object variables needed for forward and back propagation
        self.z = None
        self.a = None
        self.z_prev = None

    def activate(self):
        """
        activation function
        sigmoid for standard dense layer
        :return: sigmoid(a)
        """
        return sigmoid(self.a)

    def forward(self, z_prev):
        """
        forward propagate layer
        :param z_prev: output of the previous layer
        :return: output of the layer
        """

        # store for back prop
        self.z_prev = z_prev

        # multiply with weight matrix and add bias
        self.a = np.dot(self.z_prev, self.weight) + self.bias

        # activation function is applied
        self.z = self.activate()

        return self.z

    def backward(self, dL_dz, alpha):
        """
        backwards propagate layer
        Compute derivatives of Loss with respect to weights, biases and
        previous layers
        :param dL_dz: gradient of Loss with respect to output of this layer
        :param alpha: learning rate
        :return: gradient of Loss with respect to previous layer
        """

        # Using the chain rule we can calculate all the derivates of Loss
        # by multiply our way to the desired derivative

        # derivatives with respect to pre-activated output

        da_dw = self.z_prev
        da_db = 1
        da_dz_prev = self.weight

        # derivative of activation function
        dz_da = derivative_sigmoid(self.a)

        # derivative of loss to pre-activated output
        dL_da = dL_dz * dz_da

        # derivative of loss to weight and bias
        dL_dw = np.outer(da_dw.T, dL_da)
        dL_db = dL_da * da_db

        # derivative of loss to previous layer

        # to calculate the gradient with respect to the previous layer we
        # to use the chain rule and multiply the derivative AND in addition we
        # have to sum up all the derivatives, a neuron has to the previous
        # layer
        # a handy way, of doing this is using matrix multiplication,
        # which first multiplies the matching row/column and then sums
        # up the results

        dL_dz_prev = np.dot(dL_da, da_dz_prev.T)

        # update weights and biases with derivatives multiplied by
        # the learning rate
        self.weight -= alpha * dL_dw
        self.bias -= alpha * dL_db

        return dL_dz_prev


class LastLayer(Layer):
    """

    Last Layer in Net.
    activation = softmax

    hast to be defined differently because of derivative of softmax
    """

    def activate(self):
        """
        activation function
        softmax
        :return: softmax(a)
        """
        return softmax(self.a)

    def backward(self, dL_dz, alpha):
        """
        The derivative of the cross-entropy cost function is 0 for every
        neuron except the one of the labe.
        That's why we only have to compute the deriva
        back propagate only for output[label], because all other
        gradients are zero anyway,

        :param dL_dz: gradient of Loss with respect to output of this layer
        :param alpha: learning rate
        :return: gradient of Loss with respect to previous layer
        """

        for i, grad in enumerate(dL_dz):

            if grad != 0:
                # The derivative of the softmax function is jacobian
                # matrix. the derivative of every value is dependent on all
                # other values. if we multiply this with the derivative of
                # the Loss function, only the diagonal elements remain (all
                # other get 0), Therefore we only need to compute those to get
                # the vector with the derivatives

                # derivative of Loss with respect to pre-activated output

                y = np.zeros(10)
                y[i] = 1
                dL_da = self.z - y

                # rest is same as standard layer
                da_dw = self.z_prev
                da_db = 1
                da_dz_prev = self.weight

                # derivative of loss to weight, bias and previous layer
                dL_dw = np.outer(da_dw.T, dL_da)
                dL_db = dL_da * da_db
                dL_dz_prev = da_dz_prev @ dL_da

                # update weights and biases with learning rate alpha
                self.weight -= alpha * dL_dw
                self.bias -= alpha * dL_db

                return dL_dz_prev


def split_data(data, eval_fraction=0.1):
    """
    split data into training and validation set
    :param eval_fraction: fraction of the data to be used for evaluation
    :return: train_data, eval_data
    """

    # shuffle data
    random.shuffle(data)

    # len of evaluation data
    len_eval = int(len(data) * eval_fraction)

    train_data = data[:-len_eval]
    eval_data = data[-len_eval:]

    return train_data, eval_data


def sigmoid(a):
    """
    sigmoid activation function
    :return: 1 / (1 + np.exp(-a))
    """
    return 1 / (1 + np.exp(-a))


def derivative_sigmoid(a):
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


def cross_entropy(output, label):
    """
    cross entropy loss
    :return: -np.log(output[label])
    """
    return -np.log(output[label])


def derivative_cross_entropy(output, label):
    """
    derivative of cross entropy loss
    :return: gradient[label] = -1 / output[label]
    """
    y = np.zeros(10)
    y[label] = -1 / output[label]

    return y


def mse(output, label):
    """
    mean square error
    :return: np.sum((y - output) ** 2)
    """
    y = np.zeros(10)
    y[label] = 1
    return np.sum((y - output) ** 2)


def derivative_mse(output, label):
    """
    derivative of mse
    :return: -(y - output)
    """
    y = np.zeros(10)
    y[label] = 1
    return -(y - output)
