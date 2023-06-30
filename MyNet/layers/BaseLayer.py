# Project   : IG-Bot
# File      : BaseLayer.py
# Author    : Christoph Klösch
# Year      : 2023


class BaseLayer:
    """
    """

    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        self.input = input
        self.output = self.input

        return self.output

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, dL_dout, alpha):
        return dL_dout
