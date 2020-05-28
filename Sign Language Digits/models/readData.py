import numpy as np

def readFile():
    x = np.load('data/Sign-language-digits-dataset/X.npy') # shape 2062, 64, 64
    y = np.load('data/Sign-language-digits-dataset/Y.npy') # shape 2062, 10

    return x, y