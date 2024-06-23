import numpy as np
import activation_function

class ReLU():
    def __init__(self):
        self.mask = None
        self.out = None
    
    def forward(self, x):
        '''
        Input : nparray
        Output : ReLU(x)
        '''
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        self.out = out
        return out
    
    def backward(self, dout):
        '''
        Input : delta out
        Output : dx
        '''
        dout[self.mask] = 0
        dx = dout.copy()
        return dx
    
class sigmoid():
    
    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = activation_function.sigmoid(x)
        return self.out
    
    def backward(self, dout):
        y = self.out
        return dout*y*(1.0 - y)