import matplotlib.pyplot as plt
import numpy as np

'''
Show a grid of imagine from the dataset
        :param
            images: array of images
            label: array of label
        :return none

'''


def showImages(images, label):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel("Sign: {}".format(label[i]))
    plt.savefig('image\imgGrid.png')


'''
Show accumulative acc and loss from CNN model
        :param
            history: dictionary of the result of cnn aplication
            score: array of the evaluation
        :return none
'''


def showGraph(history, score):
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["val_acc"], label="Validation Accuracy", c="red", linewidth=4)
    plt.plot(history.history["acc"], label="Accuracy", c="green", linewidth=4)
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(history.history["val_loss"], label="Validation Loss", c="red", linewidth=4)
    plt.plot(history.history["loss"], label="Loss", c="green", linewidth=4)
    plt.legend()

    plt.savefig('images/CNN/accuracy-loss.png')


'''
Show the graph between error train and error validation
        :param
            errTrain: dictionary of the result of cnn aplication
            errVal: array of the evaluation
            m: num of evaluation
        :return none
'''


def learningCurve(errTrain, errVal, m):
    plt.plot(np.arange(1, m + 1), errTrain, label="Train")
    plt.plot(np.arange(1, m + 1), errVal, label="Validation")
    plt.show()
    plt.savefig('image/MCC/LearningCurve.png')


'''
Show the graph between error train and error validation
        :param
            errTrain: dictionary of the result of cnn aplication
            errVal: array of the evaluation
            lam: array of lambda
        :return none
'''


def lambdaError(errTrain, errVal, lam):
    fig, ax = plt.subplots()

    plt.plot(lam, errTrain, '-o', label="Train")
    plt.plot(lam, errVal, '-o', label="Validation")

    ax.set(xlabel='lambda',
           title='Comparison between ErrVal- ErrTrain')
    plt.legend()
    plt.show()
    plt.savefig('image/MCC/FinBestLambda.png')


'''
Show the cost value
        :param
            cost: array of the cost
            m: num of evaluation
        :return none
'''


def costGraph(cost, m, fig=0):
    fig, ax = plt.subplots()
    plt.plot(np.arange(1, m + 1), cost, label="Cost")
    ax.set(title='Cost Values')
    plt.legend()
    plt.show()
    plt.savefig('image/CostGraph{}.png'.format(fig))
