import numpy as np

def mse(y, y_pred):
    return np.mean(0.5 * (y - y_pred)**2)