import numpy as np
import scipy.optimize as opt
from models import utils

'''
Saving the cost while applying the mineralization function

'''
cost_history = []

'''
This function apply the cost function to x and y of the logistical regression multi-class, first apply the function
and then save the cost in the history.
    :param
        theta: weights
        x: features 
        y: labels of the features
        lam: learning rate
        
    :return
        reg_cost: it is the result from apply the cost function

'''


def cost_function(theta, x, y, lam=0.1):
    m = len(y)
    h = utils.sig_function(x @ theta)

    # calculate reg cost
    cost = (1 / m) * (-y @ np.log(h) - ((1 - y) @ np.log(1 - h)))
    reg_cost = cost + lam / 2 / m * np.sum(theta[1:] ** 2)

    # update cost history
    cost_history.append(reg_cost)
    return reg_cost


'''
This function apply the regression function to x and y of the logistical regression multi-class.

    :param
        theta: weights
        x: features 
        y: labels of the features
        lam: learning rate
        
    :return
        reg_grad: the result from apply the regression function

'''


def grad_function(theta, x, y, lam=0.1):
    m = len(y)
    h = utils.sig_function(x @ theta)
    # calculate gradient
    reg_grad = (1 / m) * (x.T @ (h - y)) + ((lam / m) * np.hstack(([0], theta[1:])))
    return reg_grad


'''
Trains num_labels logistic regression classifiers and returns each of these classifiers in a matrix all_theta, where the i-th
row of all_theta corresponds to the classifier for label i.
    :param
        x: features 
        y: labels of the features
        lam: learning rate
        
    :return
        reg


'''


def apply(x, y, lam=0.03):

    Lout = x.shape[1]
    reg = np.zeros(shape=(utils.numLabel, Lout))

    for i in range(0, utils.numLabel):
        theta0 = np.zeros(shape=(Lout, 1))
        reg[i, :] = oneVsAll(x, y, i, theta0, lam)
    return reg


'''
Run minimize to obtain the optimal theta, each numLabel is compared to the y label´s to check in which values we are.
     :param
        theta: initial weight, it contains 0 values
        numLabel: the i label 
        x: features 
        y: labels of the features
        lam: learning rate
     :return
        reg: the minimize theta for the numLabel


'''


def oneVsAll(x, y, numLabel, theta, lam):
    label = (y == numLabel).astype(int)
    reg = opt.fmin_l_bfgs_b(func=cost_function, x0=theta, fprime=grad_function, args=(x, label, lam))
    return reg[0]


'''
Predicts for each example of x.
    :param
        theta: weights
        x: features
        
    :return
        prediction: array of the prediction for x


'''


def predOneVsAll(theta, x):
    prediction = utils.sig_function(x @ theta.T)
    prediction2 = np.argmax(prediction, axis=1)
    return prediction2
