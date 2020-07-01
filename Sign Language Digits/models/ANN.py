import numpy as np
from models import utils
from scipy.optimize import minimize

'''
Saving the cost while applying the mineralization function

'''
cost_history = []

def backwardPropagation(params_ns, inputSize, hiddenSize, numLabel, x, y, lam):
    theta1 = params_ns[:((inputSize + 1) * hiddenSize)].reshape(hiddenSize, inputSize + 1)
    theta2 = params_ns[((inputSize + 1) * hiddenSize):].reshape(numLabel, hiddenSize + 1)

    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x

    m = x.shape[0]

    a1, z2, a2, a3, h = forwardPropagation(x, theta1, theta2)
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t]  # (1, 10)
        d3t = ht - yt  # (1, 10)
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))  # (1, 26)
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 = 1 / m * delta1
    delta2 = 1 / m * delta2

    delta1Reg = delta1 + (lam / m) * np.hstack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    delta2Reg = delta2 + (lam / m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))

    deltaVec = np.concatenate((delta1Reg.ravel(), delta2Reg.ravel()))
    # Calculate Cost

    reg_cost = cost_function(theta1=theta1, theta2=theta2, x=x, y=y, a=h)

    return reg_cost, deltaVec


'''
    explain forward propagation
    
'''


def forwardPropagation(x, theta1, theta2):
    # First Input Layer: Activation a(1)
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x

    a1 = X  # <a1_shape>=(1443, 4096)
    z2 = a1 @ theta1.T  # <z3_shape>=(63, 4096)
    a2 = utils.sig_function(z2)  # <a2_shape>=(63, 4096)

    aux2 = np.ones(shape=(a2.shape[0], a2.shape[1] + 1))
    aux2[:, 1:] = a2

    a3 = aux2 @ theta2.T  # <a3_shape>=(10, 4096)
    h = utils.sig_function(a3)  # <h_shape>=(10, 4096)

    return a1, z2, aux2, a3, h


def cost_function(theta1, theta2, x, y, a, lam=1):
    m = x.shape[0]

    J = (-1 / m) * np.sum((np.log(a) @ y.T) + np.log(1 - a) @ (1 - y.T))
    reg_cost = J + lam / (2 * m) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

    cost_history.append(reg_cost)
    return reg_cost


def backPropagationLearning(x, y, params_ns, hiddenSize, numLabel, inputSize, lam=1):
    fmin = minimize(fun=backwardPropagation,
                    x0=params_ns,
                    args=(inputSize, hiddenSize, numLabel, x, y, lam),
                    method='L-BFGS-B',
                    jac=True,
                    )
    return fmin
