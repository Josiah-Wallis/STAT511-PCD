import numpy as np
from scipy.optimize import brentq

def S(beta, gamma): 
    return np.sign(beta) * np.maximum(np.abs(beta) - gamma, 0)

# Fused Lasso 1D Single Predictor Objective
def objective_i(beta, betas, y, lambda1, lambda2, i):
    y_i = y[i]
    t1 = (1/2) * (y_i - beta) ** 2
    t2 = lambda1 * np.abs(beta)

    if i == 0:
        t3 = lambda2 * np.abs(betas[i + 1] - beta)
    elif i == len(betas) - 1:
        t3 = lambda2 * np.abs(beta - betas[i - 1])
    else:
        t3 = lambda2 * (np.abs(beta - betas[i - 1]) + np.abs(betas[i + 1] - beta))

    return t1 + t2 + t3

def objective_gamma_i(gamma, betas, y, lambda1, lambda2, i):
    # Assume i > 1
    y_i = y[i]
    t1 = (1/2) * (y_i  - betas[i]) ** 2
    t2 = lambda1 * np.abs(betas[i])

    if i == 0:
        t3 = lambda2 * np.abs(betas[i + 1] - betas[i])
    elif i == len(betas) - 1:
        t3 = lambda2 * gamma
    else:
        t3 = lambda2 * (gamma + np.abs(betas[i + 1] - betas[i - 1] - gamma))

    return t1 + t2 + t3


# Gradient
def dbeta_i_dc(beta ,betas, y, lambda1, lambda2, i):
    y_i = y[i]
    beta_i = beta
    beta_im1 = 0 if i == 0 else betas[i - 1]
    beta_ip1 = 0 if i == (len(betas) - 1) else betas[i + 1]

    d = -(y_i - beta_i) + lambda1 * np.sign(beta_i) - lambda2 * np.sign(beta_ip1 - beta_i) + lambda2 * np.sign(beta_i - beta_im1)

    return d

def dgamma_i_fc(gamma, betas, y, lambda1, lambda2, i):
    # Assume i > 1
    y_im1 = y[i - 1]
    y_i = y[i]
    beta_ip1 = 0 if i == (len(betas) - 1) else betas[i + 1]
    beta_im2 = betas[i - 2]

    d = -(y_im1 - gamma) - (y_i - gamma) + 2 * lambda1 * np.sign(gamma) - lambda2 * np.sign(beta_ip1 - gamma) + lambda2 * np.sign(gamma - beta_im2)

    return d

# Descent Cycle Helpers
def create_intervals_dc(betas, i):
    beta_n1 = None if i == 0 else betas[i - 1]
    beta_n2 = None if i == (len(betas) - 1) else betas[i + 1]

    if (beta_n1 is not None) and (beta_n2 is not None) and (beta_n1 > beta_n2):
        beta_n1, beta_n2 = beta_n2, beta_n1
    
    intervals = []

    intervals.append([-1e10, 0])
    if beta_n1 is None:
        intervals.append([0, beta_n2])
        intervals.append([beta_n2, 1e10])
        return intervals
    else:
        intervals.append([0, beta_n1])

    if beta_n2 is None:
        intervals.append([beta_n1, 1e10])
    else:
        intervals.append([beta_n1, beta_n2])
        intervals.append([beta_n2, 1e10])

    return intervals

def find_best_candidate(betas, y, lambda1, lambda2, i):
    intervals = create_intervals_dc(betas, i)
    
    candidates = np.array([])
    acv = []                    # active constraint value
    for a, b in intervals:
        if not (a == -1e10 or b == 1e10):
            acv.append(a)
            acv.append(b)

        try:
            candidate = brentq(dbeta_i_dc, a, b, args = (betas, y, lambda1, lambda2, i))
        except ValueError:
            candidate = None
        candidates = np.append(candidates, candidate)

    # grab unique values
    acv = list(set(acv))
    vals = []
    if np.all(candidates == None):
        for beta in acv:
            vals.append(objective_i(beta, betas, y, lambda1, lambda2, i))
        return acv[np.argmin(vals)]
    else:
        for beta in candidates:
            if beta == None:
                vals.append(1e10)
                continue
            vals.append(objective_i(beta, betas, y, lambda1, lambda2, i))
        return candidates[np.argmin(vals)]