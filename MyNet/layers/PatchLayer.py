# Project   : 
# File      : PatchLayer.py
# Author    : Christoph Klösch
# Year      : 2023
import numpy as np


class PatchLayer:
    """"""

    def __init__(self, kernel_size):
        """
        Constructor takes as input the size of the kernel
        """
        self.kernel_size = kernel_size

    def patches_generator(self, image):
        """
        Divide the input image in patches to be used during pooling.
        Yields the tuples containing the patches and their coordinates.
        """
        # Compute the ouput size
        output_h = image.shape[0] // self.kernel_size
        output_w = image.shape[1] // self.kernel_size
        self.image = image

        for h in range(output_h):
            for w in range(output_w):
                patch = image[(h * self.kernel_size):(
                        h * self.kernel_size + self.kernel_size),
                        (w * self.kernel_size):(
                                w * self.kernel_size + self.kernel_size)]
                yield patch, h, w


class MaxPoolingLayer(PatchLayer):

    def forward(self, image):
        image_h, image_w, num_kernels = image.shape
        max_pooling_output = np.zeros((image_h // self.kernel_size,
                                       image_w // self.kernel_size,
                                       num_kernels))
        for patch, h, w in self.patches_generator(image):
            max_pooling_output[h, w] = np.amax(patch, axis=(0, 1))
        return max_pooling_output

    def backward(self, dE_dY,alpha):
        """
        Takes the gradient of the loss function with respect to the output and computes the gradients of the loss function with respect
        to the kernels' weights.
        dE_dY comes from the following layer, typically softmax.
        There are no weights to update, but the output is needed to update the weights of the convolutional layer.
        """
        dE_dk = np.zeros(self.image.shape)
        for patch, h, w in self.patches_generator(self.image):
            image_h, image_w, num_kernels = patch.shape
            max_val = np.amax(patch, axis=(0, 1))

            for idx_h in range(image_h):
                for idx_w in range(image_w):
                    for idx_k in range(num_kernels):
                        if patch[idx_h, idx_w, idx_k] == max_val[idx_k]:
                            dE_dk[h * self.kernel_size + idx_h, w * self.kernel_size + idx_w, idx_k] = dE_dY[h, w, idx_k]
            return dE_dk
