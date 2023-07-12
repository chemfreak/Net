from .BaseLayer import BaseLayer

from ..utils.activations import *


# inherit from base class Layer
class ActivationLayer(BaseLayer):
    def __init__(self, activation):
        super().__init__()

        activations = {
            "tanh": (
                tanh, tanh_prime),
            "sigmoid": (sigmoid, sigmoid_prime),
            "softmax": (softmax, softmax_prime),
            "relu": (relu, relu_prime),
        }

        # set loss function and its derivative
        self.activation, self.activation_prime = activations.get(activation)

    # returns the activated input
    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, dL_dout, alpha):
        return self.activation_prime(self.input) * dL_dout


class SoftMaxLayer(ActivationLayer):
    def __init__(self):
        super(SoftMaxLayer, self).__init__("softmax")

    def backward(self, dL_dout, alpha):
        return dL_dout
