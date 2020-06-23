from models import processData
from models import LR
from data import showData as sh
import warnings

warnings.filterwarnings(action='ignore')

x, y = processData.readFile()
y = processData.createLabel(y)

xTrain, xTest, yTrain, yTest = processData.createTrain_TestData(x, y)
xTrainFlatten = processData.flatten(xTrain)
xTestFlatten = processData.flatten(xTest)
xTrainFinal = processData.normalizePixel(xTrainFlatten)
xTestFinal = processData.normalizePixel(xTestFlatten)


reg1 = LR.apply(xTrainFlatten, yTrain)
print(reg1)
#reg2 = logisticRegression.apply(xTrainFinal, yTrain)
#print(reg2)

cost_history = LR.cost_history
#print("Cost History: {}".format(cost_history))
predTest1 = LR.predOneVsAll(reg1, xTestFinal)
predTrain1 = LR.predOneVsAll(reg1, xTrainFinal)
print("Test Pred1: {}\r\nTrain Pred1: {}\r\n".format(predTest1, predTrain1))

#predTest2, predTrain2 = logisticRegression.logisticRegressionWithSklearn(xTrainFlatten, yTrain, xTestFlatten, yTest)
#print("Test Pred2: {}\r\nTrain Pred2: {}\r\n".format(predTest2, predTrain2))




