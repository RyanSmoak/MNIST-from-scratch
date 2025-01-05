import numpy as np

class Dropout:
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
            return x * self.mask
        else:
            return x  # During inference, no dropout is applied

    def backward(self, dout):
        return dout * self.mask