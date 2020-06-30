from tensorflow import keras
from tensorflow.keras import layers
from models import utils


def TwoLayerCNN(xTrain, yTrain, xVal, yVal, xTest, yTest, epoch=15):
    model = keras.Sequential(
        [
            keras.Input(shape=utils.inputShape),
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(utils.numLabel, activation="softmax")
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(xTrain, yTrain, epochs=epoch, validation_split=0.1, validation_data=(xVal, yVal))
    score = model.evaluate(xTest, yTest, verbose=0)
    print("Train accuracy of the model: ", score[1])
    print("Train loss of the model: ", score[0])
    return h, score


def TreeLayerCNN(xTrain, yTrain, xVal, yVal, xTest, yTest, epoch=15):
    model = keras.Sequential(
        [
            keras.Input(shape=utils.inputShape),
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
            layers.Dense(utils.numLabel, activation="softmax")
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(xTrain, yTrain, epochs=epoch, validation_split=0.1, validation_data=(xVal, yVal))
    score = model.evaluate(xTest, yTest, verbose=0)
    print("Train accuracy of the model: ", score[1])
    print("Train loss of the model: ", score[0])
    return h, score


def FourthLayerCNN(xTrain, yTrain, xVal, yVal, xTest, yTest, epoch=15):
    model = keras.Sequential(
        [
            keras.Input(shape=utils.inputShape),
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
            layers.Dense(utils.numLabel, activation="softmax")
        ]
    )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(xTrain, yTrain, epochs=epoch, validation_split=0.1, validation_data=(xVal, yVal))
    score = model.evaluate(xTest, yTest, verbose=0)
    print("Train accuracy of the model: ", score[1])
    print("Train loss of the model: ", score[0])
    return h, score
