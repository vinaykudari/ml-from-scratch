import numpy as np

def to_poly(X, degree=2):
    Xp = np.ones((X.shape[0], degree))
    for i in range(degree):
        Xp[:, i] = np.power(X, i+1).ravel()
    return Xp