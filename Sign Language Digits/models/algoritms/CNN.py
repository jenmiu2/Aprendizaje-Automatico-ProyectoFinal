from tensorflow import keras
from tensorflow.keras import layers
from models import utils
from models import showData

inputShape = (64, 64, 1)
'''
'''


def twoLayerCNN(xTrain, yTrain, xTest, yTest, epoch=100):
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
            layers.Dense(utils.numLabel, activation="softmax")
        ]
    )
    applyModel(model, xTrain, xTest, yTrain, yTest, epoch=epoch)


'''
'''


def fourthLayerCNN(xTrain, yTrain, xTest, yTest, epoch=100):
    model = keras.Sequential(
        [
            keras.Input(shape=inputShape),
            layers.Conv2D(filters=8, kernel_size=(3, 3), padding="Same", activation="relu"),
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
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(utils.numLabel, activation="softmax")
        ]
    )
    applyModel(model, xTrain, xTest, yTrain, yTest, epoch=epoch)


def applyModel(model, xTrain, xTest, yTrain, yTest, epoch=100):
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(xTrain, yTrain, epochs=epoch, validation_split=0.1)
    score = model.evaluate(xTest, yTest, verbose=0)
    showData.showGraph(h)
    showData.showMapHeat(model.predict(xTest), yTest)
    print("Train accuracy of the model: ", h.history['acc'][-1] * 100)
    print("Test loss of the model: ", score[0])
    print("Test accuracy of the model: ", score[1] * 100)
