import numpy as np
import scipy.optimize as opt
from sklearn import linear_model

cost_history = []
def cost_function(theta, x, y, lam=1):
    res = x @ theta
    h = sig_function(res)

    error = -y * np.log(h) - ((1 - y) * np.log(1 - h))
    cost = (1 / len(y)) * sum(error)
    reg_cost = cost + lam / (2 * len(y)) * sum(theta[1:] ** 2)
    cost_history.append(reg_cost)
    return reg_cost


def grad_function(theta, x, y, lam=3):
    m = len(y)
    h = sig_function(x @ theta)
    reg_grad = (1 / m) * (x.T @ (h - y)) + ((lam / m) * theta)

    return reg_grad


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s


def oneVsAll(x, y, num_etiquetas, theta):
    label = (y == num_etiquetas).astype(int)
    reg = opt.fmin_tnc(func=cost_function, x0=theta, fprime=grad_function, args=(x, label))[1]
    return reg


def predOneVsAll(theta, x):
    prediction = sig_function(x @ theta.T)
    prediction = np.argmax(prediction, axis=1) + 1
    return prediction


def logisticRegressionWithSklearn(xTrain, yTrain, xTest, yTest, max_iter=500):
    logReg = linear_model.LogisticRegression(random_state=1, max_iter=max_iter)
    predTest = logReg.fit(xTrain, yTrain).score(xTest, yTest)
    predTrain = logReg.fit(xTrain, yTrain).score(xTrain, yTrain)
    return predTest, predTrain
