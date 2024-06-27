from layer import *
from functions import numerical_grad
from collections import OrderedDict

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            'W1':2 * np.random.randn(input_size, hidden_size) / np.sqrt(input_size),
            'b1':np.zeros(hidden_size),
            'W2':2 * np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size),
            'b2':np.zeros(output_size)
        }
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affain(self.params['W1'], self.params['b1'])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affain(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {
            'W1':numerical_grad(loss_W, self.params['W1']),
            'b1':numerical_grad(loss_W, self.params['b1']),
            'W2':numerical_grad(loss_W, self.params['W2']),
            'b2':numerical_grad(loss_W, self.params['b2']),
        }
        return grads
    
    def gradient(self, x, t):
        #순전파
        self.loss(x, t)
        
        #역전파
        dout = 1
        dout = self.lastLayer.backward()
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {
            'W1':self.layers['Affine1'].dW,
            'b1':self.layers['Affine1'].db,
            'W2':self.layers['Affine2'].dW,
            'b2':self.layers['Affine2'].db
        }

        return grads