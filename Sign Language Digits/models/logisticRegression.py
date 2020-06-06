import numpy as np
import scipy.optimize as opt
from sklearn import linear_model

cost_history = []


def cost_function(theta, x, y, lam=1):
    th = theta[:(x.shape[1] * y.shape[1])].reshape(x.shape[1], y.shape[1])
    h = sig_function(x @ th)

    error = -y * np.log(h) - ((1 - y) * np.log(1 - h))
    cost = (1 / len(y)) * sum(error)
    reg_cost = cost + lam / (2 * len(y)) * sum(theta[1:] ** 2)
    cost_history.append(reg_cost[0])
    return reg_cost[0]


def grad_function(theta, x, y, lam=3):
    m = len(y)
    th = theta[:(x.shape[1] * y.shape[1])].reshape(x.shape[1], y.shape[1])
    h = sig_function(x @ th)
    reg_grad = (1 / m) * (x.T @ (h - y)) + ((lam / m) * th)
    ravel = reg_grad.ravel()
    return ravel


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s


def apply(x, y):
    numEt = 10
    Lout = x.shape[1]
    reg = np.zeros(shape=(numEt, numEt * x.shape[1]))

    for i in range(0, numEt):
        theta0 = np.zeros(shape=(Lout, y.shape[1]))
        reg[i, :] = oneVsAll(x, y, theta0)
    return reg


def oneVsAll(x, y, theta):
    reg = opt.fmin_tnc(func=cost_function, x0=theta, fprime=grad_function, args=(x, y))
    return reg[1]


def predOneVsAll(theta, x):
    prediction = sig_function(x @ theta.T)
    prediction = np.argmax(prediction, axis=1) + 1
    return prediction


def logisticRegressionWithSklearn(xTrain, yTrain, xTest, yTest, max_iter=500):
    logReg = linear_model.LogisticRegression(random_state=1, max_iter=max_iter)
    predTest = logReg.fit(xTrain, yTrain).score(xTest, yTest)
    predTrain = logReg.fit(xTrain, yTrain).score(xTrain, yTrain)
    return predTest, predTrain
