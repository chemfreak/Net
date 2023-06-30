from .BaseLayer import BaseLayer

from ..activations import softmax, softmax_dummy_prime


# inherit from base class Layer
class ActivationLayer(BaseLayer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

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
        super(SoftMaxLayer, self).__init__(softmax, softmax_dummy_prime)

    def backward(self, dL_dout, alpha):
        print("dL_dout", dL_dout)
        return dL_dout
