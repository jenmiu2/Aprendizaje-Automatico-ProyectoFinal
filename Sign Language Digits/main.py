from models import processData
from models import logisticRegression
import warnings
import numpy as np
import tensorflow as tf

warnings.filterwarnings(action='ignore')

x, y = processData.readFile()
y = processData.createLabel(y)
xTrain, xTest, yTrain, yTest = processData.createTrain_TestData(x, y)
xTrainFlatten = processData.flatten(xTrain)
xTestFlatten = processData.flatten(xTest)
xTrainFinal = processData.normalizePixel(xTrainFlatten)
xTestFinal = processData.normalizePixel(xTestFlatten)


reg1 = logisticRegression.apply(xTrainFlatten, yTrain)
print(reg1)
#reg2 = logisticRegression.apply(xTrainFinal, yTrain)
#print(reg2)

cost_history = logisticRegression.cost_history
#print("Cost History: {}".format(cost_history))
predTest1 = logisticRegression.predOneVsAll(reg1, xTestFinal)
predTrain1 = logisticRegression.predOneVsAll(reg1, xTrainFinal)
print("Test Pred1: {}\r\nTrain Pred1: {}\r\n".format((np.mean(np.abs(predTest1 - yTest)) * 100), (np.mean(np.abs(predTrain1 - yTest)) * 100)))

#predTest2, predTrain2 = logisticRegression.logisticRegressionWithSklearn(xTrainFlatten, yTrain, xTestFlatten, yTest)
#print("Test Pred2: {}\r\nTrain Pred2: {}\r\n".format(predTest2, predTrain2))




