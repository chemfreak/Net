import numpy as np

from MyNet import Net
from network import Network
from MyNet.layers import DenseLayer, ActivationLayer, SoftMaxLayer, ConvLayer
from MyNet.activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from MyNet.losses import mse, mse_prime

# training data
x_train = [np.random.rand(10, 10, 1)]

y_train = [np.random.rand(4, 4, 2)]


data = list(zip(x_train, y_train))

layers = [ConvLayer((10, 10, 1), 3, 1),
          ActivationLayer(tanh, tanh_prime),
          ConvLayer((8, 8, 1), 3, 1),
          ActivationLayer(tanh, tanh_prime),
          ConvLayer((6, 6, 1), 3, 2),
          ActivationLayer(tanh, tanh_prime)
          ]
# network
net = Net(layers, loss="mse")

# train

net.fit(data, data, epochs=1000, alpha=0.3)

# test
out = net.predict(x_train)
print("predicted = ", out)
print("expected = ", y_train)
