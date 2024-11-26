import numpy as np

def generate_data(n, p):
    X = np.random.randint(0, 10, size = (n, p))
    beta_true = np.random.randint(1, 100, size = p)
    bias_true = np.random.randint(10, 20)
    y = X @ beta_true + bias_true + np.random.standard_normal(n)

    return X, y, beta_true, bias_true

def l2_scale(X):
    X_centered = X - np.mean(X, axis = 0)
    X_norm = X_centered / np.linalg.norm(X_centered, axis = 0)

    return X_norm