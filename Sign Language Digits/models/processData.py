import numpy as np
from sklearn.model_selection import train_test_split
import math as mt
import warnings

from models import utils

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


def createTrain_TestData(x, y, grade=0.30):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, stratify=y, test_size=grade, random_state=1)
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
    :returns
        matrix: random matrix
'''


def randomWeight(Lin, Lout):
    matrix = (np.random.random(size=Lin * (Lout + 1) +
                                    utils.numLabel * (Lin + 1)) - 0.5) * 0.25
    return matrix


'''
The function create the real number of a onehot encoder type.
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
This function read the data from the file and separate each type in Train data,
Validation data and Test data, then flatten the X type data and finally it normalizes the pixel.
    :param none
    :returns
        xTrainFinal: the training data
        xTestFinal: the test data
        yTrain: the training label
        yTest: the test label
        y: the transformation data of the y label into real number
    
'''


def createValTrainTest():
    x, y_hot = readFile()  # shape =(2062, 64, 64), shape =(2062, 10)
    y = createLabel(y_hot)  # shape =(2062, )

    xTrain, xTest, yTrain, yTest = createTrain_TestData(x, y)
    # xTrain, xVal, yTrain, yVal = createTrain_TestData(x, y, grade=0.5)

    xTrainFlatten = flatten(xTrain)  # grad shape =(1443, 4096)
    xTestFlatten = flatten(xTest)  # shape =(619, 4096)
    xTrainFinal = normalizePixel(xTrainFlatten)  # shape =(1443, 4096)
    xTestFinal = normalizePixel(xTestFlatten)  # shape =(619, 4096)

    return xTrainFinal, xTestFinal, yTrain, yTest, y_hot
