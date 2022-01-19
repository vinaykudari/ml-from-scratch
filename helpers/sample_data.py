import numpy as np
from sklearn.datasets import make_classification

def linear_data(
    n_samples,
    n_features,
    noise=10,
    seed=0,
):
    np.random.seed(seed)
    coef = [np.random.randint(0, 9)]
    x = np.random.rand(n_samples, n_features)
    y = coef[-1] * x 
    
    return x, y, coef


def non_linear_data(
    n_samples,
    n_features,
    noise=0,
    seed=0,
    degree=3
):
    np.random.seed(seed)
    coef = []
    y = 0
    x = np.random.rand(n_samples, n_features)
    
    for i in range(degree, -1, -1):
        coef.append(np.random.randint(-degree, degree))
        y += coef[-1] * (x ** i)
    
    return x, y, coef


def classification_data(
    n_samples,
    n_features=3,
    n_classes=3,
    seed=0,
):
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=n_classes,
        flip_y=0,
        random_state=seed,
    )
    
    return x, y


