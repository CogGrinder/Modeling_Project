import numpy as np
import cv2
import matplotlib.pyplot as plt

from image import Image

img = Image("images/clean_finger.png")
img.display()

img.rotate(-16, (0.7, 0.5))

img.display()