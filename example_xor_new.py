import numpy as np

from MyNet import Net
from network import Network
from MyNet.layers import DenseLayer, ActivationLayer,SoftMaxLayer
from MyNet.activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from MyNet.losses import mse, mse_prime

# training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

data = list(zip(x_train, y_train))

layers = [DenseLayer(2, 3),
          ActivationLayer(tanh, tanh_prime),
          DenseLayer(3, 2),
          ActivationLayer(tanh, tanh_prime),
          ]

net = Net(layers,loss="mse")

net.fit(data, data, alpha=0.1, epochs=1000)
# test
out = net.predict(x_train)
print(out)
exit()
# network
net = Network()
net.add(DenseLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(DenseLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
