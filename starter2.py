import numpy as np
import cv2
import matplotlib.pyplot as plt

from image import Image

img = Image("images/clean_finger.png")
img.display()

img.rotate_translate(-16, (100, 100), (0,0))

img.display()
