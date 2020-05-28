import numpy as np
import scipy.optimize as opt


def cost_function(theta, x, y, lam=1):
    h = sig_function(x @ theta)

    h[h == 1] = 0.999
    error = -y * np.log(h) - ((1 - y) * np.log(1 - h))
    cost = (1 / len(y)) * sum(error)
    reg_cost = cost + lam / (2 * len(y)) * sum(theta[1:] ** 2)

    return reg_cost[0]


def grad_function(theta, x, y, lam=3):
    m = len(y)
    h = sig_function(x @ theta)
    # para j = 0
    grad_0 = (1 / m) * (x.T @ (h - y))[0]

    # para j >= 1
    grad_1 = (1 / m) * (x.T @ (h - y))[1:] + ((lam / m) * theta[1:])

    reg_grad = np.insert(grad_1, 0, grad_0)
    return reg_grad


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s


def oneVsAll(x, y, num_etiquetas, reg, theta):
    label = (y == num_etiquetas).astype(int)
    reg[num_etiquetas, :] = opt.fmin_tnc(func=cost_function, x0=theta, fprime=grad_function, args=(x, label))[1]
    return reg


def predOneVsAll(theta, x):
    prediction = sig_function(x @ theta.T)
    prediction = np.argmax(prediction, axis=1) + 1
    return prediction