import numpy as np
from models import utils
from scipy.optimize import minimize

'''
Saving the cost while applying the mineralization function

'''
cost_history = []

'''
This function is the representation of backward propagation applying foward propagation and
the calculate of the gradient.
    :param
        x: array of features
        y: array of labels in real format
        params_ns: theta1 and theta2 concatenate
        hiddenSize: num of hidden layer
        numLabel: different types of labels
        inputSize: num of the first layer
        lam: lambda
    :return
        reg_cost: cost
        deltaVec: concatenate the weights

'''


def backwardPropagation(params_ns, inputSize, hiddenSize, numLabel, x, y, lam):
    theta1 = params_ns[:((inputSize + 1) * hiddenSize)].reshape(hiddenSize, inputSize + 1)  # <theta1_shape>=(23, 4097)
    theta2 = params_ns[((inputSize + 1) * hiddenSize):].reshape(numLabel, hiddenSize + 1)  # <theta2_shape>=(10, 24)

    delta1 = np.zeros(theta1.shape)  # <delta1_shape>=(23, 4097)
    delta2 = np.zeros(theta2.shape)  # <delta2_shape>=(10, 24)

    m = x.shape[0]

    a1, a2, h = forwardPropagation(x, theta1, theta2)
    for t in range(m):
        a1t = a1[t, :]  # <a1t_shape>=(4097)
        a2t = a2[t, :]  # <a2t_shape>=(24)
        ht = h[t, :]  # <ht_shape>=(10)
        yt = y[t]  # int
        d3t = ht - yt  # <a2_shape>=(10, )
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))  # <d2t_shape>=(24, )
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = 1 / m * delta1
    delta2 = 1 / m * delta2

    delta1Reg = delta1 + (lam / m) * np.hstack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    delta2Reg = delta2 + (lam / m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))

    deltaVec = np.concatenate((delta1Reg.ravel(), delta2Reg.ravel()))  # <deltaVec_shape>=(94471)
    # Calculate Cost

    reg_cost = cost_function(y=y, a=h)

    return reg_cost, deltaVec


'''
This function is the representation of foward propagation apply for each layer the sigmoid function.
    :param
        x: array of label
        theta1: first layer weight
        theta2: second layer weight
    :return
        a1: first activation
        aux2: second activation
        h: result activation

'''


def forwardPropagation(x, theta1, theta2):
    # First Input Layer: Activation a(1)
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x

    a1 = X  # <a1_shape>=(1443, 4097)
    z2 = a1 @ theta1.T  # <z2_shape>=(1443, 23)
    a2 = utils.sig_function(z2)  # <a2_shape>=(1443, 23)

    # Second Input Layer: Activation a(2)
    aux2 = np.ones(shape=(a2.shape[0], a2.shape[1] + 1))
    aux2[:, 1:] = a2

    # Third Input Layer: Activation a(3)
    a3 = aux2 @ theta2.T  # <a3_shape>=(1443, 10)
    h = utils.sig_function(a3)  # <h_shape>=(1443, 10)

    return a1, aux2, h


'''
This is the implementation of the cross entropy function.
    :param
        y: array of label
        a: activation
    :return 
        cost: applying cost function to y and a

'''


def cost_function(y, a):
    m = y.shape[0]

    error = y @ np.log(a)
    cost = -np.sum(error) / m

    cost_history.append(cost)
    return cost


'''
This function minimize the backward propagation with L-BFGS-B algorithm
    :param
        x: array of features
        y: array of labels in real format
        params_ns: theta1 and theta2 concatenate
        hiddenSize: num of hidden layer
        numLabel: different types of labels
        inputSize: num of the first layer
        lam: lambda
    :return
        fmin: minimization of the algorithm 

'''


def backPropagationLearning(x, y, params_ns, hiddenSize, numLabel, inputSize, lam=1):
    fmin = minimize(fun=backwardPropagation,
                    x0=params_ns,
                    args=(inputSize, hiddenSize, numLabel, x, y, lam),
                    method='L-BFGS-B',
                    jac=True)
    return fmin['x']
