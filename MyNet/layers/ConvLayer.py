from .BaseLayer import BaseLayer
from scipy import signal
import numpy as np


## Math behind this layer can found at :
## https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

# inherit from base class Layer
# This convolutional layer is always with stride 1
class ConvLayer(BaseLayer):
    # input_shape = (i,j,d)
    # kernel_shape = (m,n)
    # layer_depth = output_depth
    def __init__(self, input_shape, kernel_size=3, layer_depth=8):
        super().__init__()
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = (kernel_size, kernel_size)
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0] - self.kernel_shape[0] + 1,
                             input_shape[1] - self.kernel_shape[1] + 1,
                             layer_depth)
        self.weights = np.random.rand(self.kernel_shape[0],
                                      self.kernel_shape[1], self.input_depth,
                                      layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    # returns output for a given input
    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:, :, k] += signal.correlate2d(
                    self.input[:, :, d], self.weights[:, :, d, k], 'valid') + \
                                        self.bias[k]

        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward(self, output_error, alpha):
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros(self.weights.shape)
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:, :, d] += signal.convolve2d(output_error[:, :, k],
                                                       self.weights[:, :, d,
                                                       k], 'full')
                dWeights[:, :, d, k] = signal.correlate2d(self.input[:, :, d],
                                                          output_error[:, :,
                                                          k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_error[:, :, k])

        self.weights -= alpha * dWeights
        self.bias -= alpha * dBias
        return in_error
