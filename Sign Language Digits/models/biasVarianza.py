import numpy as np
from models import modelSelection as ms
from models import showData


def findBestLambda(x, y, xVal, yVal):
    lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    m = len(lams)
    errTrain, errVal = np.zeros(m), np.zeros(m)
    theta = np.ones(x.shape[1])

    for i, lam in enumerate(lams):
        fmin = ms.minGradient(theta, x, y, lam=lam)['x']
        errTrain[i] = ms.LinearGradienteCost(fmin, x, y, lam=0)[0]
        errVal[i] = ms.LinearGradienteCost(fmin, xVal, yVal, lam=0)[0]

    showData.parte3_2(errTrain, errVal, lams)


def learningCurve(x, y, xVal, yVal, lam):
    m = len(x)
    errTrain, errVal = np.zeros(m), np.zeros(m)
    theta = np.ones(x.shape[1])

    for i in range(1, m + 1):
        x_c = x[:i]
        y_c = y[:i]

        fmin = ms.minGradient(theta, x_c, y_c, lam=lam)['x']

        errTrain[i - 1] = ms.LinearGradienteCost(fmin, x_c, y_c, lam=0)[0]
        errVal[i - 1] = ms.LinearGradienteCost(fmin, xVal, yVal, lam=0)[0]
    showData.parte2(errTrain, errVal, m)
    return errTrain, errVal
