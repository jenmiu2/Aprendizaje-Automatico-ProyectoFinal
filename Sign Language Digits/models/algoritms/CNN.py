from tensorflow import keras
from tensorflow.keras import layers
from models import utils




def twoLayerCNN(xTrain, yTrain, xTest, yTest, epoch=100):
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
    h, score = applyModel(model, xTrain, yTrain, xTest, yTest, epoch=epoch)
    return h, score



def fourthLayerCNN(xTrain, yTrain, xTest, yTest, epoch=100):
    model = keras.Sequential(
        [
            keras.Input(shape=utils.inputShape),
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
    h, score = applyModel(model, xTrain, yTrain, xTest, yTest, epoch=epoch)
    return h, score


def applyModel(model, xTrain, xTest, yTrain, yTest, epoch=100):
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    h = model.fit(xTrain, yTrain, epochs=epoch, validation_split=0.1)
    score = model.evaluate(xTest, yTest, verbose=0)
    print("Train accuracy of the model: ", h.history['acc'][-1] * 100)
    print("Test loss of the model: ", score[0])
    print("Test accuracy of the model: ", score[1] * 100)
    return h, score
