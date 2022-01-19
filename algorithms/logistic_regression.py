# add parent directory to path: enable import from parent dir
import math
import sys
sys.path.append('../')

import numpy as np
from matplotlib import pyplot as plt

from helpers.sample_data import classification_data
from helpers.loss_functions import mse
from helpers.utils.activation_functions import softmax
from helpers.utils.transforms import to_one_hot


class LogisticRegression:
    def __init__(
        self,
        epochs=500,
        lr=0.1,
    ):
        self.epochs = epochs
        self.lr = lr
    
    def _gradient(self, data, Y):
        self.Z = - data.dot(self.params)
        gradient = 1/self.n_samples * (data.T @ (Y - softmax(self.Z))) + 2 * self.params
        return gradient
        
    def _init_params(self, n_features, n_classes):
        # xavier weight initialization
        x_range = 1 / (math.sqrt(n_features))
        self.params = np.random.uniform(
            -x_range,
            x_range,
            (n_features, n_classes)
        )
        
    def train(self, data, labels):
        # add bias
        data = np.c_[np.ones(data.shape), data]
        y = to_one_hot(labels)
        self.n_classes = y.shape[1]
        self.n_samples, self.n_features = data.shape
        self._init_params(self.n_features, self.n_classes)
        
        for iteration in range(self.epochs):
            self.params -= self.lr * self._gradient(data, y)
            
            
    def predict(self, X):
        # add bias
        X = np.c_[np.ones(X.shape), X]
        preds = softmax(-X.dot(self.params))
        return np.argmax(preds, axis=1)