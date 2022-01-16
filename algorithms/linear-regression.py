import sys
sys.path.append('../')

import math
import numpy as np
from matplotlib import pyplot as plt

from helpers.sample_data import linear_data
from helpers.loss_functions import mse

class LinearRegression:
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
    ):
        self.epochs = epochs
        self.lr = lr
        self.losses = []
    
    def _init_params(self, n_features):
        # xavier weight initialization
        x_range = 1 / (math.sqrt(n_features))
        self.params = np.random.uniform(
            -x_range,
            x_range,
            (n_features, )
        )
        
    def train(self, data, labels):
        # add bias
        data = np.c_[np.ones(data.shape), data]
        self.n, self.n_features = data.shape
        self._init_params(self.n_features)
        
        for iteration in range(self.epochs):
            preds = data.dot(self.params)
            loss = mse(labels, preds)
            self.losses.append(loss)
            grad_params = -(labels - preds).dot(data)
            self.params -= self.lr * grad_params
            
            
    def predict(self, X):
        # add bias
        X = np.c_[np.ones(X.shape), X]
        preds = X.dot(self.params)
        return preds