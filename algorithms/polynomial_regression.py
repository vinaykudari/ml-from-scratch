import sys
sys.path.append('../')

from helpers.utils.transforms import to_poly
from algorithms.linear_regression import LinearRegression


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