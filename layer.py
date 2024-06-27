import numpy as np
import activation_function
from functions import *

class ReLU():
    def __init__(self):
        self.mask = None
        self.out = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        self.out = out
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Sigmoid():
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = activation_function.sigmoid(x)
        return self.out
    
    def backward(self, dout):
        y = self.out
        return dout*y*(1.0 - y)

class Affain():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.out = None

    def forward(self, x):
        self.x = x
        self.out = np.dot(x, self.W) + self.b
        return self.out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = activation_function.softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    