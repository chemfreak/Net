from .BaseLayer import BaseLayer
import numpy as np


# inherit from base class Layer
class DenseLayer(BaseLayer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        super().__init__()

        self.weights = np.random.rand(input_size,
                                      output_size) / input_size
        # print(self.weights)
        self.bias = np.random.rand(1, output_size)

    # returns output for a given input
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        # print(self.output)
        return self.output

    def backward(self, dL_dout, alpha):
        """
        backwards propagate layer
        Compute derivatives of Loss with respect to weights, biases and
        previous layers
        :param dL_dout: gradient of Loss with respect to output of this layer
        :param alpha: learning rate
        :return: gradient of Loss with respect to previous layer
        """

        # Using the chain rule we can calculate all the derivates of Loss
        # by multiply our way to the desired derivative

        # derivatives with respect to output

        dout_dw = self.input
        dout_db = 1
        dout_din = self.weights

        # derivative of loss to weight and bias
        dL_dw = np.outer(dout_dw.T, dL_dout)
        dL_db = dL_dout * dout_db

        # derivative of loss to previous layer

        # to calculate the gradient with respect to the previous layer we
        # to use the chain rule and multiply the derivative AND in addition we
        # have to sum up all the derivatives, a neuron has to the previous
        # layer
        # a handy way, of doing this is using matrix multiplication,
        # which first multiplies the matching row/column and then sums
        # up the results

        dL_din = np.dot(dL_dout, dout_din.T)

        # update weights and biases with derivatives multiplied by
        # the learning rate
        self.weights -= alpha * dL_dw
        self.bias -= alpha * dL_db

        return dL_din
