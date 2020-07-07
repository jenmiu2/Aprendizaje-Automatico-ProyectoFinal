import numpy as np
from sklearn.model_selection import train_test_split
from models import utils
import warnings

warnings.filterwarnings(action='ignore')

'''
This function read the files adding to x the columns of ones 
    :param none
    :returns 
        x: contain the data image, shape of x (2062, 64, 64) 
        y: contain the label of training, shape of y (2062, 10)
'''


def readFile():
    x = np.load('data/Sign-language-digits-dataset/X.npy')
    y = np.load('data/Sign-language-digits-dataset/Y.npy')
    return x, y


'''
 It flatten the data from 3D array into 2D
    :param
        x: the 3D´s array
    :return
        xFlatten: the x transformation into an array of 2 dimension
    
'''


def flatten(x):
    size = x.shape[0]
    areaPixel = x.shape[1] * x.shape[2]
    xFlatten = x.reshape(size, areaPixel)
    return xFlatten


'''
We separate the test data from the train data, because we want to 
know if the model understands the
difference with new values. Each time we use this function, the train
data and the test data will be difference.

    :param
        x: image data
        y: label data
        grade: default value 30%, the size of the  
    :returns
        xTrain: training data from the parameter x, if we use the default value it´s
                going to be the 70 % of the total of x
        xTest: test data from the parameter x, if we use the default value it´s
                going to be the 70 % of the total of y
        yTrain: training data from the parameter x, if we use the default value it´s
                going to be the 30 % of the total of x
        yTest: test data from the parameter x, if we use the default value it´s
                going to be the 30 % of the total of y
'''


def createTrain_TestData(x, y, stratify=True, grade=0.30):
    if stratify:
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, stratify=y, test_size=grade, random_state=20)
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=grade, random_state=20)
    return xTrain, xTest, yTrain, yTest


'''
Normalize the pixel of each data to be between 0 and 1, we want to speed up the process of deep learning.
    :param 
        x: image data
        y: label data
    :returns
        xNormalize: image data normalize between 0 and 1
        yNormalize: label data normalize between 0 and 1
        
'''


def normalizePixel(x):
    xNormalize = x / 255.0
    return xNormalize


'''
The weights of each layer of the neural network are initialized
    :param
        Lin: number of columns
        Lout: number of rows
        ram: decide if the initialization is going to be random or zeros
    :returns
        matrix: random matrix
'''


def randomWeight(Lin, Lout, ram=True):
    if ram:
        matrix = (np.random.random(size=Lin * (Lout + 1) +
                                        utils.numLabel * (Lin + 1)) - 0.5) * 0.25
    else:
        matrix = np.zeros((Lin * (Lout + 1) + utils.numLabel * (Lin + 1), 1))
    return matrix


'''
The function create the real number of a one-hot encoder type.
    :param
        y: label encode on one hot
    :return
        y_label: the transformation of the y label into real number
'''


def createLabel(y):
    y_new = [np.argmax(target) for target in y]
    y_label = np.array(y_new)
    return y_label


'''
The function fix the label order, the given order is: 9, 0, 7, 6, 1, 8, 4, 3, 2, 5
    :param
        y: label encode real
    :return
        y_label: fix labeling
'''


def correctLabel(y):
    label = {0: 9, 1: 0, 2: 7, 3: 6, 4: 1, 5: 8, 6: 4, 7: 3, 8: 2, 9: 5}
    y_new = [label[target] for target in y]
    y_label = np.array(y_new)
    return y_label


'''
This function read the data from the file and separate each type in Train data,
Validation data and Test data, then flatten the X type data and finally it normalizes the pixel.
    :param 
        val: default false, indicates if you want an extra partition for the validation data
        flatNorm: default true, indicates if you want flatten the array and normalize the pixel
        realNumber: default false, indicates if you want the conversion from one hot to real number
    :returns
        xTrainFinal: the training data
        xTestFinal: the test data
        yTrain: the training label
        yTest: the test label
        y: the transformation data of the y label into real number
    
'''


def createValTrainTest(val=False, flatNorm=True, realNumber=False):
    x, y = readFile()  # <shape x> =(2062, 64, 64), <shape y> =(2062, 10)

    if flatNorm:
        x = flatten(x)  # <shape x> =(2062, 4096)
        x = normalizePixel(x)  # <shape x[1]> =(2062, 4096)
    if realNumber:
        y = createLabel(y)  # <shape y> =(2062, )
        y = correctLabel(y)

    # <shape[0] xTrain> =(1443), <shape[0] xTest> =(619) <shape[0] yTrain> =(2062), <shape[0] yTest> =(619 )
    xTrain, xTest, yTrain, yTest = createTrain_TestData(x, y, stratify=flatNorm)

    if val:
        # <shape[0] xTrain> =(1226), <shape[0] xVal> =(217), <shape[0] yTrain> =(1226), <shape[0] yVal> =(217)
        xTrain, xVal, yTrain, yVal = createTrain_TestData(xTrain, yTrain, grade=0.15)
        return xTrain, xVal, xTest, yTrain, yVal, yTest

    return xTrain, xTest, yTrain, yTest
