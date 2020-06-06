from models import processData
from models import logisticRegression
import warnings

warnings.filterwarnings(action='ignore')

x, y = processData.readFile()

xTrain, xTest, yTrain, yTest = processData.createTrain_TestData(x, y)
xTrainFlatten = processData.flatten(xTrain)
xTestFlatten = processData.flatten(xTest)
xTrainFinal = processData.normalizePixel(xTrainFlatten)
xTestFinal = processData.normalizePixel(xTestFlatten)


#reg = logisticRegression.apply(xTrainFlatten, yTrain)
reg = logisticRegression.apply(xTrainFinal, yTrain)
print(reg)

cost_history = logisticRegression.cost_history
print("Cost History: {}".format(cost_history))
#predTest1 = logisticRegression.predOneVsAll(reg, xTestFinal)
#predTrain1 = logisticRegression.predOneVsAll(reg, xTrainFinal)
#print("Test Pred1: {}\r\nTrain Pred1: {}\r\n".format(predTest1, predTrain1))

#predTest2, predTrain2 = logisticRegression.logisticRegressionWithSklearn(xTrainFlatten, yTrain, xTestFlatten, yTest)
#print("Test Pred2: {}\r\nTrain Pred2: {}\r\n".format(predTest2, predTrain2))




