import matplotlib.pyplot as plt
import random as rd
import numpy as np

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
    plt.savefig('Img.png')
