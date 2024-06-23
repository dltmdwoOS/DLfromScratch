import numpy as np

def step_function(x):
    if x>0:
        return 1
    else:
        return 0

def sigmoid(x): # elementwise
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return x if x > 0 else 0

def softmax(x): # elementwise
    exp_x = np.exp(x)
    exp_x_sum = np.sum(exp_x)
    return exp_x / exp_x_sum
