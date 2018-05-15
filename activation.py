import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)

def relu(x):
    return np.maximum(x, 0)

def drelu(x):
    a = np.copy(x)
    for i in np.nditer(a, op_flags=['readwrite']):
        if i < 0:
            i[...] = 0
        else:
            i[...] = 1
    return a

def softmax(x, t = 1.0):
    e = np.exp(np.array(x) / t)
    dist = e / np.sum(e)
    return dist

