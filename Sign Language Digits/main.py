from models import processData
from models import MultiClassClassification
from models import CNN
from models import ANN
from models import utils
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

'''
This function create the result of the multiply regression class, 
for that we need the train data and the test
data, then we apply the oneVsAll to create the theta´s os each x and y

'''


def applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest):
    reg = MultiClassClassification.apply(xTrain, yTrain)  # shape =(10, 4096)

    cost_history = MultiClassClassification.cost_history
    print(np.min(cost_history))

    predTest = MultiClassClassification.predOneVsAll(reg, xTest)  # shape =(619, )
    predTrain = MultiClassClassification.predOneVsAll(reg, xTrain)  # shape =(1443, )

    print("Probability Test: {}%\r\nProbability Train: {}%\r\n "
          .format(np.mean(predTest == yTest) * 100,
                  np.mean(predTrain == yTrain) * 100))
    return cost_history, reg


def applyCNN(xTrain, xTest, xVal, yTrain, yTest, yVal):
    epochs = [15, 25, 40, 70, 100]
    x_train = np.expand_dims(xTrain, -1)
    x_val = np.expand_dims(xVal, -1)
    x_test = np.expand_dims(xTest, -1)


def applyANN(xTrain, xTest, yTrain, yTest):
    inputSizeTrain = xTrain.shape[0]  # 1443
    hiddenSize = 63  # random

    theta0 = processData.randomWeight(hiddenSize, inputSizeTrain)
    theta1 = processData.randomWeight(inputSizeTrain, hiddenSize)

    params_ns = np.concatenate((theta0.ravel(), theta1.ravel()))
    fmin = ANN.backPropagationLearning(xTrain, yTrain, params_ns, hiddenSize, utils.numLabel, inputSizeTrain)

    print(fmin)

    return fmin.fun, fmin['x']


def selectBestAlgorithm():
    xTrain, xTest, yTrain, yTest = processData.createValTrainTest()
    # MCLR
    cost_historyMCLR, predMCLR = applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest)

    # ANN
    cost_historyANN, predANN = applyANN(xTrain, xTest, yTrain, yTest)

    # CNN
    cost_historyCNN, predCNN = applyCNN(xTrain, xTest, yTrain, yTest)
