import numpy as np
import scipy.optimize as opt

cost_history = []


def cost_function(theta, x, y, lam=1):
    h = sig_function(x @ theta)

    error = -y * np.log(h) - ((1 - y) * np.log(1 - h))
    cost = (1 / len(y)) * sum(error)
    reg_cost = cost + lam / (2 * len(y)) * sum(theta[1:] ** 2)
    cost_history.append(reg_cost)
    return reg_cost


def grad_function(theta, x, y, lam=3):
    m = len(y)
    h = sig_function(x @ theta)
    reg_grad = (1 / m) * (x.T @ (h - y)) + ((lam / m) * theta)
    ravel = reg_grad.ravel()
    return ravel


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s


def apply(x, y):
    maxLabel = 10
    Lout = x.shape[1]
    reg = np.zeros(shape=(maxLabel, x.shape[1]))

    for i in range(0, maxLabel):
        theta0 = np.zeros(shape=(Lout, 1))
        reg[i, :] = oneVsAll(x, y, i, theta0)
    return reg


def oneVsAll(x, y, numLabel, theta):
    label = (y == numLabel).astype(int)
    reg = opt.fmin_tnc(func=cost_function, x0=theta, fprime=grad_function, args=(x, label))
    return reg[1]


def predOneVsAll(theta, x):
    prediction = sig_function(x @ theta.T)
    prediction2 = np.argmax(prediction, axis=1) + 1
    return prediction2
