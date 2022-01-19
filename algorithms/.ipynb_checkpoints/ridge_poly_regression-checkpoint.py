import sys
sys.path.append('../')

from helpers.utils.transforms import to_poly
from helpers.utils.regularization import RidgeRegularization
from algorithms.polynomial_regression import PolynomialRegression


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