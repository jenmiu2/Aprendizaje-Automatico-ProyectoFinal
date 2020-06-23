import numpy as np
from sklearn.model_selection import train_test_split
import math as mt
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
3D -> 2D
'''
def flatten(x):
    size = x.shape[0]
    areaPixel = x.shape[1] * x.shape[2]
    xTrainFlatten = x.reshape(size, areaPixel)
    return xTrainFlatten


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


def randomWeight(Lin, Lout, e=0.12):
    e = mt.sqrt(6) / mt.sqrt(Lin + Lout)
    matrix = np.random.random(size=(Lout, Lin))
    matrix = matrix * 2 * e
    matrix = matrix - e
    return matrix


'''
We need to create an array with the values on number instead of on-hot format.

'''


def createLabel(y):
    y_new = [np.argmax(target) for target in y]
    array = np.array(y_new)
    return array

