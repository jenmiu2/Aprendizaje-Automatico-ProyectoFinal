import numpy as np
from models.biasVarianza import modelSelection as ms
from models import showData

'''
The function show the comparison graph between the error created from evaluate train data
and validation data with different values of lambda. First, minimize the theta value and then store
the result of the minimization to the values.
        :param
            x: array of train data
            y: array of train label
            xVal: array of validation data
            yval: array of validation label
            showGraph: indicates if it is show the plot or print the data in the console
        :return none 

'''


def findBestLambda(x, y, xVal, yVal, lams, m, showGraph=True):
    errTrain, errVal = np.zeros(m), np.zeros(m)
    theta = np.ones(x.shape[1])

    for i, lam in enumerate(lams):
        fmin = ms.minGradient(theta, x, y, lam=lam)['x']
        errTrain[i] = ms.linearGradienteCost(fmin, x, y, lam=0)[0]
        errVal[i] = ms.linearGradienteCost(fmin, xVal, yVal, lam=0)[0]

    if showGraph:
        showData.lambdaError(errTrain, errVal, lams)
    else:
        print("Error validation: {}\n\rError Testing: {}".format(np.min(errVal), np.min(errTrain)))


'''
The function show the comparison graph between the error created from evaluate train data
and validation data applying the given lambda.
        :param
            x: array of train data
            y: array of train label
            xVal: array of validation data
            yval: array of validation label
        :return none 

'''


def learningCurve(x, y, xVal, yVal, lam):
    m = len(x)
    errTrain, errVal = np.zeros(m), np.zeros(m)
    theta = np.ones(x.shape[1])

    for i in range(1, m + 1):
        x_c = x[:i]
        y_c = y[:i]

        fmin = ms.minGradient(theta, x_c, y_c, lam=lam)['x']

        errTrain[i - 1] = ms.linearGradienteCost(fmin, x_c, y_c, lam=0)[0]
        errVal[i - 1] = ms.linearGradienteCost(fmin, xVal, yVal, lam=0)[0]
    showData.learningCurve(errTrain, errVal, m)
