import numpy as np

epochs = [15, 25, 40, 70, 100]
numLabel = 10
inputShape = (64, 64, 1)

'''
This function is the derivative of the sigmoid function
    :param
        x: array of features to apply the function

    :return
        s: sigmoid application in x

'''


def sig_dev_function(x):
    s = 1 / (1 + np.exp(-x))
    s = s * (1 - s)
    return s


'''
This function is the sigmoid function
    :param
        x: array of features to apply the function

    :return
        s: sigmoid application in x

'''


def sig_function(x):
    s = 1 / (1 + np.exp(-x))
    return s
