import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
'''
Show a grid of imagenes from the datasheet

'''


def showImages(images, label, figsize=10, tamGrid=5):
    plt.figure(figsize=(figsize, figsize))
    for i in range(25):
        plt.subplot(tamGrid, tamGrid, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel("Sign: {}".format(label[i]))
    plt.savefig('image\imgGrid.png')


'''
Show acummulative acc and loss from CNN model

'''


def showGraph(history, score, epochs, fase):
    plt.figure(figsize=(24, 8))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["val_acc"], label="Validation Accuracy", c="red", linewidth=4)
    plt.plot(history.history["acc"], label="Accuracy", c="green", linewidth=4)
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(history.history["val_loss"], label="Validation Loss", c="red", linewidth=4)
    plt.plot(history.history["loss"], label="Loss", c="green", linewidth=4)
    plt.legend()

    # Generate the name of the Graph
    name = "Acc-Loss Fase{}-Epochs{}.png".format(fase, epochs)
    plt.savefig('image\CNN\{}'.format(name))

'''
errTraing frente a errVal
'''
def parte2(errTrain, errVal, m, fig):
    plt.plot(np.arange(1, m + 1), errTrain, label="Train")
    plt.plot(np.arange(1, m + 1), errVal, label="Validation")
    plt.show()
    plt.savefig('image/fig2-{}.png'.format(fig), dpi=300)

'''

err -> lambda
'''

def parte3_2(errTrain, errVal, lam):
    plt.figure()
    plt.plot(lam, errTrain, '-o', label="Train")
    plt.plot(lam, errVal, '-o', label="Validation")
    plt.legend()
    plt.show()
    plt.savefig('image/fig3-2.png', dpi=300)
