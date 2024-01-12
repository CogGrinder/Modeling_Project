import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

from image import Image

img = Image("images/moist_finger.png")
img.display()
img.symmetry()
img.display()