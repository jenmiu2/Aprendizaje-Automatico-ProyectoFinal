from models import processData
from models import logisticRegression
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

Lin = 1
Lout = 4096
num_etiquetas = 10
x, y = processData.readFile()


xTrain, xTest, yTrain, yTest = processData.createTrain_TestData(x, y)
xTrainFlatten = processData.flatten(xTrain)
xTestFlatten = processData.flatten(xTest)

theta = processData.randomWeight(Lin, Lout)

cost = logisticRegression.cost_function(theta, xTrainFlatten, yTrain)
grad = logisticRegression.grad_function(theta, xTrainFlatten, yTrain)
print("Cost: {}\r\nReg: {}\r\n".format(cost, grad))


reg = np.zeros(shape=(num_etiquetas, x.shape[1]))
for i in range(0, num_etiquetas):
    theta0 = np.zeros(shape=(Lout, 1))
    reg[i, :] = logisticRegression.oneVsAll(xTrainFlatten, yTrain, i, theta0)


predTest, predTrain = logisticRegression.logisticRegressionWithSklearn(xTrainFlatten, yTrain, xTestFlatten, yTest)
print("Test Pred: {}\r\nTrain Pred: {}\r\n".format(predTest, predTrain))




