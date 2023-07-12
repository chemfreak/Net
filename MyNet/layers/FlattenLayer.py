# Project   : 
# File      : FlattenLayer.py
# Author    : Christoph Klösch
# Year      : 2023


# inherit from base class Layer
from MyNet.layers import BaseLayer


class FlattenLayer(BaseLayer):
    # returns the flattened input
    def forward(self, input_data):
        #print(input_data.shape)
        self.input = input_data
        self.output = input_data.flatten().reshape(1, -1)
        #print(self.output.shape)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward(self, output_error, alpha):
        return output_error.reshape(self.input.shape)
