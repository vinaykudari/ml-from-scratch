import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    n = np.exp(x - np.max(x))
    d = np.sum(n, axis=1)
    return n/np.expand_dims(d, 1)