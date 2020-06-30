from models import processData
from models import MultiClassClassification
from models import CNN
from models import ANN
from models import showData as sd
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

'''
This function create the result of the multiply regression class, 
for that we need the train data and the test
data, then we apply the oneVsAll to create the theta´s os each x and y

'''
def applyMultipleRegressionClass():

    xTrain, xTest, yTrain, yTest, _ = processData.createValTrainTest()
    reg1 = MultiClassClassification.apply(xTrain, yTrain)  # shape =(10, 4096)

    cost_history = MultiClassClassification.cost_history
    print(np.min(cost_history))

    predTest = MultiClassClassification.predOneVsAll(reg1, xTest)  # shape =(619, )
    predTrain = MultiClassClassification.predOneVsAll(reg1, xTrain)  # shape =(1443, )


    print("Probability Test: {}%\r\nProbability Train: {}%\r\n "
          .format(np.mean(predTest == yTest) * 100,
                  np.mean(predTrain == yTrain) * 100))


def applyCNN():
    '''
    # 2-Layer
    for i in CNN.epochs:
        h, score = CNN.TwoLayerCNN(epoch=i)
        sd.showGraph(history=h, score=score, epochs=i, fase=2)
     # 3-Layer
    for i in CNN.epochs:
        h, score = CNN.TwoLayerCNN(epoch=i)
        sd.showGraph(history=h, score=score, epochs=i, fase=3)
    '''
    # 4-Layer

    h, score = CNN.FourthLayerCNN(epoch=100)
    sd.showGraph(history=h, score=score, epochs=100, fase=4)


def applyANN():
    xTrain, xTest, yTrain, yTest, _ = processData.createValTrainTest()

    inputSizeTrain = xTrain.shape[0]# 1443
    hiddenSize = 63# random
    numLabel = 10

    theta0 = processData.randomWeight(hiddenSize, inputSizeTrain)
    theta1 = processData.randomWeight(inputSizeTrain, hiddenSize)

    params_ns = np.concatenate((theta0.ravel(), theta1.ravel()))
    fmin = ANN.backPropagationLearning(xTrain, yTrain, params_ns, hiddenSize, numLabel, inputSizeTrain)

    print(fmin)


applyMultipleRegressionClass()
