import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

from image import Image

# kernel = np.ones((15, 15))/225
img = Image("images/clean_finger.png")
img.conv_2d(150,150)
img.display()