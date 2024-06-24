import numpy as np

# Loss functions
def mean_square_error(y, t):
    batch_size = y.shape[0]
    return np.sum((y-t)**2) / (2*batch_size)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = np.reshape(1, y.size)
        t = np.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_grad(f, x):
    h = 1e-4
    grad = np.zeros(x.shape)
    for idx in range(x.shape):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fx1 = f(x)
        x[idx] = tmp_val - h
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*h)
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_grad(f, x)
        x -= lr*grad
    return x