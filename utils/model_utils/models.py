import numpy as np
from .cd_helpers import S

def lasso_cd(X_unscaled, y, reg_param, tol = 1e-5):
    from ..test_utils import l2_scale

    _, p = X_unscaled.shape
    beta_s = np.random.randn(p)

    X = l2_scale(X_unscaled)

    while True:
        beta_sp1 = np.copy(beta_s)

        for j in range(p):
            resids = y - (X @ beta_sp1)
            sum_term = X[:, j] @ resids.reshape(-1, 1)
            beta_sp1[j] = S(sum_term[0], reg_param)
        
        if np.max(np.abs(beta_sp1 - beta_s)) < tol:
            break

        beta_s = beta_sp1

    scaling = np.linalg.norm(X_unscaled - np.mean(X_unscaled, axis = 0), axis = 0)
    beta = beta_sp1 / scaling

    return beta