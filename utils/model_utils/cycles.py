import numpy as np
from scipy.optimize import brentq
from .cd_helpers import find_best_candidate, objective_i, objective_gamma_i, dgamma_i_fc

def descent_cycle(p, y, lambda1, lambda2, tol = 1e-5):
    beta_s = np.zeros(p)
    beta_sp1 = np.copy(beta_s)

    while True:

        for j in range(p):
            prev_err = objective_i(beta_sp1[j], beta_sp1, y, lambda1, lambda2, j)
            beta_sp1[j] = find_best_candidate(beta_sp1, y, lambda1, lambda2, j)
            curr_err = objective_i(beta_sp1[j], beta_sp1, y, lambda1, lambda2, j)
            
            if (j >= 2) and (curr_err - prev_err > 0):
                print('test')
                fusion_cycle(beta_sp1, y, lambda1, lambda2, j)

        if np.max(np.abs(beta_sp1 - beta_s)) < tol:
            break
        
        beta_s = np.copy(beta_sp1)

    return beta_sp1

def fusion_cycle(betas, y, lambda1, lambda2, i):

    prev_err = objective_gamma_i(betas[i] - betas[i - 1], betas, y, lambda1, lambda2, i)
    gamma = brentq(dgamma_i_fc, -1e10, 1e10, args = (betas, y, lambda1, lambda2, i))
    curr_err = objective_gamma_i(gamma, betas, y, lambda1, lambda2, i)

    if curr_err - prev_err > 0:
        betas[i] = gamma
        betas[i - 1] = gamma

        avg_y = (y[i - 1] + y[i]) / 2
        y[i - 1] = avg_y
        y[i] = avg_y


def smoothing_cycle():
    pass