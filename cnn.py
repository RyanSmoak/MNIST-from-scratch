import numpy as np

from activations import *
from conv import *
from dropout import *
from evaluations import *
from fully_connected import *
from loss import *
from SGD_Nestrov import *
from MaxPool2D import *
from flattening import *

class CNN():
  def __init__(self):
    self.conv1 = Conv2DLayer(input_shape=(1,28,28),
                             filter_shape = (32,1,2,2),
                             stride=1,
                             padding=1)

    self.pool1 = MaxPool2D(pool_size=(2,2), stride=2, padding=0)

    self.conv2 = Conv2DLayer(input_shape=(32,14,14),
                             filter_shape = (64,32,2,2),
                             stride=1,
                             padding=1)

    self.pool2 = MaxPool2D(pool_size=(2,2), stride=2, padding=0)
    self.flatten = Flatten()

    self.prelu_fc1 = PReLU()

    self.fc1 = FullyConnectedLayer(3136, 128,
                                   activation_function= self.prelu_fc1.forward,
                                   activation_derivative=self.prelu_fc1.backward)

    self.dropout = Dropout(keep_prob=0.5)

    self.fc2 = FullyConnectedLayer(128, 10,
                                   activation_function= Softmax().softmax,
                                   activation_derivative=None)

    self.softmax = Softmax()
    self.prelu1 = PReLU()
    self.prelu2 = PReLU()


  def forward(self, x):
    out = self.conv1.forward(x)
    out = self.prelu1.forward(out)
    out = self.pool1.forward(out)

    out = self.conv2.forward(out)
    out = self.prelu2.forward(out)
    out = self.pool2.forward(out)

    out = self.flatten.flatten(out)

    out = self.fc1.forward(out)
    out = self.prelu_fc1.forward(out)

    out = self.dropout.forward(out, training=True)

    out = self.fc2.forward(out)
    out = self.softmax.softmax(out)

    return out

  def backward(self, d_loss, learning_rate, optimizer):
    optimizer_conv1 = optimizer[0]
    optimizer_conv2 = optimizer[1]
    grad = d_loss

    grad = self.fc2.backward(grad.T, learning_rate)
    grad = self.dropout.backward(grad)

    grad = self.prelu_fc1.backward(grad)
    grad = self.fc1.backward(grad, learning_rate)
    grad = self.flatten.unflatten(grad)

    grad = self.pool2.backward(grad)
    grad = self.prelu2.backward(grad)
    grad = self.conv2.backward(grad, optimizer_conv2)

    grad = self.pool1.backward(grad)
    grad = self.prelu1.backward(grad)
    grad = self.conv1.backward(grad, optimizer_conv1)

    return grad