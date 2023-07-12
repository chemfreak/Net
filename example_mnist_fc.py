import numpy as np

from MyNet import Net
from network import Network
from MyNet.layers import DenseLayer,ActivationLayer
from MyNet.activations import tanh, tanh_prime

from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

layers= [
    DenseLayer(28*28, 100),               # input_shape=(1, 28*28)    ;
        # output_shape=(1, 100)
    ActivationLayer(tanh, tanh_prime),
    DenseLayer(100, 50),                   # input_shape=(1, 100)      ;
    # output_shape=(1, 50)
    ActivationLayer(tanh, tanh_prime),
    DenseLayer(50, 10),                  # input_shape=(1, 50)       ;
    # output_shape=(1, 10)
    ActivationLayer(tanh, tanh_prime)
]

# Network
net = Net(layers,loss="mse")

train_data = list(zip(x_train,y_train))
eval_data = list(zip(x_test,y_test))

net.fit(train_data, eval_data, epochs=10, alpha=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
