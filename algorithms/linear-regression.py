import sys
sys.path.append('../')

import math
import numpy as np
from matplotlib import pyplot as plt

from helpers.sample_data import linear_data, non_linear_data
from helpers.loss_functions import mse
from helpers.utils.transforms import to_poly
from helpers.utils.regularization import LassoRegularization, \
    RidgeRegularization, ElasticNetRegularization

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
        data = np.c_[np.ones(data.shape), data]
        self.n, self.n_features = data.shape
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
        X = np.c_[np.ones(X.shape), X]
        preds = X.dot(self.params)
        return preds
    
    
class PolynomialRegression(LinearRegression):
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
        degree=2,
    ):
        self.degree = degree
        super().__init__(epochs, lr)
        
    def train(self, data, labels):
        super().train(to_poly(data, self.degree), labels)
        
    def predict(self, X):
        return super().predict(to_poly(X, self.degree))
    
    
class LassoRegression(LinearRegression):
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
        alpha=0.001,
    ):
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.regularization = LassoRegularization(alpha=alpha)
        self.losses = []
        
    def _gradient(self, labels, preds, data):
        return -(labels - preds).dot(data) + self.regularization.grad(self.params)
    
    
class LassoPolynomialRegression(PolynomialRegression):
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
        alpha=0.001,
        degree=2,
    ):
        self.epochs = epochs
        self.lr = lr
        self.degree = degree
        self.alpha = alpha
        self.regularization = LassoRegularization(alpha=alpha)
        self.losses = []
        
    def _gradient(self, labels, preds, data):
        return -(labels - preds).dot(data) + self.regularization.grad(self.params)
    
    
class RidgeRegression(LinearRegression):
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
        alpha=0.001,
    ):
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.regularization = RidgeRegularization(alpha=alpha)
        self.losses = []
        
    def _gradient(self, labels, preds, data):
        return -(labels - preds).dot(data) + self.regularization.grad(self.params)
    
    
class RidgePolynomialRegression(PolynomialRegression):
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
        alpha=0.001,
        degree=2,
    ):
        self.epochs = epochs
        self.lr = lr
        self.degree = degree
        self.alpha = alpha
        self.regularization = RidgeRegularization(alpha=alpha)
        self.losses = []
        
    def _gradient(self, labels, preds, data):
        return -(labels - preds).dot(data) + self.regularization.grad(self.params)
    
    
class ElasticNetPolynomialRegression(PolynomialRegression):
    def __init__(
        self,
        epochs=1000,
        lr=0.0005,
        alpha=0.001,
        degree=2,
    ):
        self.epochs = epochs
        self.lr = lr
        self.degree = degree
        self.alpha = alpha
        self.regularization = ElasticNetRegularization(alpha=alpha)
        self.losses = []
        
    def _gradient(self, labels, preds, data):
        return -(labels - preds).dot(data) + self.regularization.grad(self.params)