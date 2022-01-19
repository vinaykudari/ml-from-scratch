import sys
sys.path.append('../')

from helpers.utils.regularization import RidgeRegularization
from algorithms.linear_regression import LinearRegression


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