import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

def conv_2d(image, kernel):

    # Padded version with zeros of the image
    padded_image = np.pad(image, 1, mode='constant')
    n, m = image.shape

    for i in range(0, n):
        for j in range(0, m):
            print(padded_image[i+1,j+1])


image = np.array([[1, 2, 1, 4], [1, 1, 1, 1], [2, 2, 3, 1], [1, 1, 1, 1]])
print(image)
kernel = np.array([[5, 5, 1], [6, 5, 1], [1, 2, 1]])

print(conv_2d(image, kernel))
print(signal.convolve2d(image, kernel, mode="same"))

x = np.array([[0, 0, 0], [0, 1, 2], [0, 1, 1]])
print(np.tensordot(x, kernel, axes=((0,1),(0,1))))