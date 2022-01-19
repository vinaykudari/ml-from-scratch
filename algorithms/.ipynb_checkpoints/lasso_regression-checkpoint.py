import sys
sys.path.append('../')

from helpers.utils.transforms import to_poly
from helpers.utils.regularization import LassoRegularization
from algorithms.linear_regression import LinearRegression


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