import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

from image import Image

# def conv_2d(image, kernel):

#     N = kernel.shape[0]
#     n,m = image.shape

#     # Padded version with edge of the image
#     padded_image = np.pad(image, (N-1)//2, mode='constant')

#     result_image = np.zeros((n,m))
    
#     for i in range(0, n):
#         for j in range(0, m):
#             result = 0
#             for N_i in range(-(N-1)//2, (N-1)//2 + 1):
#                 for N_j in range(-(N-1)//2, (N-1)//2 + 1):
#                     result += kernel[N_i + (N-1)//2 ][N_j + (N-1)//2] * padded_image[i-N_i][j-N_j]
#             result_image[i][j] = result
#     return result_image

# # image = np.array([[0,0,0,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]])
# image = np.array([[1, 2, 1, 4], [1, 1, 1, 1], [2, 2, 3, 1], [1, 1, 1, 1]])
# print(image)

# # kernel = np.ones((3,3))
# kernel = np.array([[5, 5, 1], [6, 5, 1], [1, 2, 1]])

# print(conv_2d(image, kernel))
# print(signal.convolve2d(image, kernel, mode="same"))

# x = np.array([[0, 0, 0], [0, 1, 2], [0, 1, 1]])
# #print(np.tensordot(x, kernel, axes=((0,1),(0,1))))

# kernel = np.ones((15, 15))/225
img = Image("images/clean_finger.png")
K = img.kernel(0,0)
img.fft_conv_2d(K)
img.display()