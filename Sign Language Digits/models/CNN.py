from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from models import processData
from tensorflow.keras import layers
import numpy as np

epochs = [15, 25, 40, 70, 100]
numLabel = 10
inputShape = (64, 64, 1)
x, y = processData.readFile()
xTrain, xTest, yTrain, yTest = processData.createTrain_TestData(x, y)  # shape =(2062, 64, 64), shape =(2062, 10)
xTrainFinal = processData.normalizePixel(xTrain)
xTestFinal = processData.normalizePixel(xTest)
x_train = np.expand_dims(xTrainFinal, -1)
x_test = np.expand_dims(xTestFinal, -1)


def TwoLayerCNN(epoch=15):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(numLabel, activation="softmax")
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(x_train, yTrain, epochs=epoch, validation_split=0.1, validation_data=(x_test, yTest))
    score = model.evaluate(x_test, yTest, verbose=0)
    print("Train accuracy of the model: ", score[1])
    print("Train loss of the model: ", score[0])
    return h, score


def TreeLayerCNN(epoch=15):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(numLabel, activation="softmax")
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(x_train, yTrain, epochs=epoch, validation_split=0.1, validation_data=(x_test, yTest))
    score = model.evaluate(x_test, yTest, verbose=0)
    print("Train accuracy of the model: ", score[1])
    print("Train loss of the model: ", score[0])
    return h, score


def FourthLayerCNN(epoch=15):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Conv2D(filters=8, kernel_size=(5, 5), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(numLabel, activation="softmax")
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(x_train, yTrain, epochs=epoch, validation_split=0.1, validation_data=(x_test, yTest))
    score = model.evaluate(x_test, yTest, verbose=0)
    print("Train accuracy of the model: ", score[1])
    print("Train loss of the model: ", score[0])
    return h, score
