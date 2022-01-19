import sys
sys.path.append('../')

import math
import numpy as np
from matplotlib import pyplot as plt

from helpers.utils.loss_functions import mse

class LinearRegression:
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
    ):
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        self.epochs = epochs
        self.lr = lr
        self.losses = []
        
    def _gradient(self, labels, preds, data):
        return -(labels - preds).dot(data)
    
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
        data = np.c_[np.ones(data.shape[0]), data]
        self.n_samples, self.n_features = data.shape
        self._init_params(self.n_features)
        
        for iteration in range(self.epochs):
            preds = data.dot(self.params)
            loss = mse(labels, preds) + self.regularization(self.params)
            self.losses.append(loss)
            gradient = self._gradient(
                labels.ravel(), 
                preds,
                data,
            )
            self.params -= self.lr * gradient
            
            
    def predict(self, X):
        # add bias
        X = np.c_[np.ones(X.shape[0]), X]
        preds = X.dot(self.params)
        return preds