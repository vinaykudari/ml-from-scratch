import numpy as np

def linear_data(
    n_samples,
    n_features,
    noise=10,
    seed=0,
):
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
    coef = []
    y = 0
    x = np.random.rand(n_samples, n_features)
    
    for i in range(degree, -1, -1):
        coef.append(np.random.randint(-degree, degree))
        y += coef[-1] * (x ** i)
    
    return x, y, coef
