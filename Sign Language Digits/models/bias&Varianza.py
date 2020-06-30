import numpy as np

from models import MultiClassClassification as mccr


def findBestLambda(x, y, xVal, yVal):
    lams = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    m = len(lams)
    errTrain, errVal = np.zeros(m), np.zeros(m)

    for i, lam in enumerate(lams):
        fmin = mccr.apply(x, y)
        errTrain[i] = mccr.cost_function(fmin, x, y, lam=0)
        errVal[i] = mccr.cost_function(fmin, xVal, yVal, lam=0)

    idx = np.argwhere(np.diff(np.sign(errTrain - errVal)) != 0).reshape(-1) + 0

    return lams[idx]


def learningCurve(x, y, xVal, yVal, lam):
    m = len(x)
    errTrain, errVal = np.zeros(m), np.zeros(m)

    for i in range(1, m + 1):
        x_c = x[:i]
        y_c = y[:i]

        fmin = mccr.apply(x, y, lam)

        errTrain[i - 1] = mccr.cost_function(fmin, x_c, y_c, lam=0)
        errVal[i - 1] = mccr.cost_function(fmin, xVal, yVal, lam=0)
    return errTrain, errVal
