from models import processData
from models.algoritms import ANN, CNN, MultiClassClassification
from models import utils
from models.biasVarianza import biasVarianza as bv
import models.showData as sd
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')



def applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest, lam=0.1, ram=False):
    reg = MultiClassClassification.apply(xTrain, yTrain, lam=lam, ram=ram)
    cost_history = MultiClassClassification.cost_history
    print(np.min(cost_history))

    predTest = MultiClassClassification.predOneVsAll(reg, xTest)
    predTrain = MultiClassClassification.predOneVsAll(reg, xTrain)

    print("Probability Test: {}%\r\nProbability Train: {}%\r\n "
          .format(np.mean(np.abs(predTest - yTest)) * 100 - 100,
                  np.mean(np.abs(predTrain - yTrain)) * 100 - 100))
    sd.costGraph(cost_history, len(cost_history), 1)


def applyCNN(xTrain, xTest, yTrain, yTest):
    x_train = np.expand_dims(xTrain, -1)
    x_test = np.expand_dims(xTest, -1)

    CNN.twoLayerCNN(x_train, yTrain, x_test, yTest, epoch=100)
   # CNN.fourthLayerCNN(x_train, yTrain, x_test, yTest, epoch=100)


def applyANN(xTrain, xTest, yTrain, yTest, lam=0.1):
    inputSizeTrain = xTrain.shape[1]  # 1443
    hiddenSize = 50  # random

    params_ns = processData.randomWeight(hiddenSize, inputSizeTrain, ram=True)
    fmin = ANN.backPropagationLearning(xTrain, yTrain, params_ns, hiddenSize, utils.numLabel, inputSizeTrain, lam)

    theta1 = fmin[:((inputSizeTrain + 1) * hiddenSize)].reshape(hiddenSize, inputSizeTrain + 1)
    theta2 = fmin[((inputSizeTrain + 1) * hiddenSize):].reshape(utils.numLabel, hiddenSize + 1)

    cost_history = ANN.cost_history
    print(np.min(cost_history))
    sd.costGraph(cost_history, len(cost_history), 1)

    a1, a2, predTest = ANN.forwardPropagation(xTest, theta1, theta2)
    a1, a2, predTrain = ANN.forwardPropagation(xTrain, theta1, theta2)

    print("Probability Test: {}%\r\nProbability Train: {}%\r\n "
          .format(np.mean(np.abs(predTest.T - yTest)) * 100 - 300,
                  np.mean(np.abs(predTrain.T - yTrain)) * 100 - 300))


def selectBestAlgorithm():

    xTrain, xVal, xTest, yTrain, yVal, yTest = processData.createValTrainTest(val=True, flatNorm=True, realNumber=True)
    print("Iniciando la busqueda del mejor lambda...")
    bv.findBestLambda(xTrain, yTrain, xVal, yVal, lams=utils.lams, m=len(utils.lams), showGraph=True)
    lam = 0.3

    print("Checking error values of lambda: {}".format(lam))
    bv.findBestLambda(xVal, yVal, xTest, yTest, lams=utils.lams, m=len(utils.lams), showGraph=False)

    print("Entrenamiento con lam = {}...".format(lam))
    bv.learningCurve(xTrain, yTrain, xVal, yVal, lam=lam)

    xTrain, xTest, yTrain, yTest = processData.createValTrainTest(val=False, flatNorm=True, realNumber=True)
    print("Multi Reg class con lam = {}...".format(lam))
    applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest, lam=lam, ram=True)
    applyMultipleRegressionClass(xTrain, xTest, yTrain, yTest, lam=lam, ram=False)

    print("ANN con lam = {}...".format(lam))
    applyANN(xTrain, xTest, yTrain, yTest, lam)

    xTrain, xTest, yTrain, yTest = processData.createValTrainTest(val=False, flatNorm=False, realNumber=False)
    print("CNN....")
    applyCNN(xTrain, xTest, yTrain, yTest)


xTrain, xTest, yTrain, yTest = processData.createValTrainTest(val=False, flatNorm=False, realNumber=False)
print("CNN....")
applyCNN(xTrain, xTest, yTrain, yTest)
