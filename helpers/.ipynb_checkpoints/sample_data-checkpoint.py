from sklearn import datasets


def linear_data(
    n_samples,
    n_features,
    noise=10,
    seed=0,
):
    x, y, coef = datasets.make_regression(
        n_samples=n_samples, 
        n_features=n_features,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=seed,
    ) 
    
    return x, y, coef