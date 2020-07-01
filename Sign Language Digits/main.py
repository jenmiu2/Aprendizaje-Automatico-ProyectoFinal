from models import processData
from models import MultiClassClassification
from models import CNN
from models import ANN
from models import utils
from models import biasVarianza as bv
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

'''
This function create the result of the multiply regression class, 
for that we need the train data and the test
data, then we apply the oneVsAll to create the theta´s os each x and y

'''


def applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest, lam=0.1):
    reg = MultiClassClassification.apply(xTrain, yTrain)
    print(reg)
    cost_history = MultiClassClassification.cost_history
    print(np.min(cost_history))

    predTest = MultiClassClassification.predOneVsAll(reg, xTest)
    predTrain = MultiClassClassification.predOneVsAll(reg, xTrain)

    print("Probability Test: {}%\r\nProbability Train: {}%\r\n "
          .format(np.mean(predTest == yTest) * 100,
                  np.mean(predTrain == yTrain) * 100))

    return cost_history, reg


def applyCNN(xTrain, xTest, xVal, yTrain, yTest, yVal):
    epochs = [15, 25, 40, 70, 100]
    x_train = np.expand_dims(xTrain, -1)
    x_val = np.expand_dims(xVal, -1)
    x_test = np.expand_dims(xTest, -1)


def applyANN(xTrain, xTest, yTrain, yTest, lam=0.1):
    inputSizeTrain = xTrain.shape[1] # 1443
    hiddenSize = 63  # random

    params_ns = processData.randomWeight(hiddenSize, inputSizeTrain)
    fmin = ANN.backPropagationLearning(xTrain, yTrain, params_ns, hiddenSize, utils.numLabel, inputSizeTrain)

    theta1 = params_ns[:((inputSizeTrain + 1) * hiddenSize)].reshape(hiddenSize, inputSizeTrain + 1)
    theta2 = params_ns[((inputSizeTrain + 1) * hiddenSize):].reshape(utils.numLabel, hiddenSize + 1)

    cost_history = ANN.cost_history

    a1, z2, aux2, a3, predTest = ANN.forwardPropagation(xTest, theta1, theta2)
    a1, z2, aux2, a3, predTrain = ANN.forwardPropagation(xTrain, theta1, theta2)

    print("Probability Test: { .4f}%\r\nProbability Train: {.4f}%\r\n "
          .format(np.mean(predTest.argmax(axis=1) == yTest) * 100,
                  np.mean(predTrain.argmax(axis=1) == yTrain) * 100))
    return cost_history, fmin


def selectBestAlgorithm():
    xTrain, xTest, yTrain, yTest, y = processData.createValTrainTest()
    lam = bv.findBestLambda(xTrain, yTrain)

    # MCLR
    cost_historyMCLR, predMCLR = applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest, lam)

    # ANN
    cost_historyANN, predANN = applyANN(xTrain, xTest, yTrain, yTest, lam)

    # CNN
    cost_historyCNN, predCNN = applyCNN(xTrain, xTest, yTrain, yTest)


xTrain, xTest, yTrain, yTest, y = processData.createValTrainTest()
cost_history, fmin = applyANN(xTrain, xTest, y, yTest)
