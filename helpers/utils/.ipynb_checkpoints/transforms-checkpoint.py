import numpy as np

def to_poly(X, degree=2):
    Xp = np.ones((X.shape[0], degree))
    for i in range(degree):
        Xp[:, i] = np.power(X, i+1).ravel()
    return Xp

def to_one_hot(X):
    m = max(X)
    res = np.zeros((X.shape[0], m+1))
    for idx, i in enumerate(X):
        res[idx][i] = 1
        
    return res