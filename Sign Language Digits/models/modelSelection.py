import numpy as np
from scipy.optimize import minimize

costHistory = []


def LinearGradienteCost(theta, x, y, lam=1):
    M = len(x)
    h = theta @ x.T

    # cost
    cost = 1 / 2 / M * np.sum((h - y)**2)
    reg_cost = cost + lam / 2 / M * np.sum(theta[1:]**2)

    # grad,  j = 0
    grad = (1 / M) * ((h - y).T @ x)

    # grad, j >= 1
    grad[1:] = grad[1:] + ((lam / M) * theta[1:])

    # update cost history

    costHistory.append(reg_cost)
    return reg_cost, grad


def error(theta, x, y):
    M = len(x)
    h = theta @ x.T
    err = 1 / 2 / M * np.sum((h - y) ** 2)
    return err


def minGradient(theta, x, y, lam):
    fmin = minimize(fun=LinearGradienteCost,
                    x0=theta,
                    args=(x, y, lam),
                    method='L-BFGS-B',
                    jac=True)

    return fmin