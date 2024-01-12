import numpy as np
import matplotlib.pyplot as plt
import math

# from image import Image
from starter2 import Starter_2

class Main_Course_1:

    @staticmethod
    def distance_between_pixels(x, y):
            ''' Return the distance between pixels x of coordinates (x1, x2) and point y of coordinates (y1, y2) 
                The distance corresponds to the distance between the centers of the two pixels '''
            # let's get the center of the pixels
            center_x = Starter_2.pixel_center(x[0], x[1])
            center_y = Starter_2.pixel_center(y[0], y[1])        
            return np.sqrt((center_x[0] - center_y[0])**2 + (center_x[1] - center_y[1])**2)
    
    @staticmethod
    def c1(r):
        return 1 / (r+1)

    @staticmethod
    def c2(r):
        return np.exp(-r)

    @staticmethod
    def c3(r):
        u = 5
        s = np.sqrt(0.2)
        return 1/2 * (1 + math.erf((r - u)/(s*np.sqrt(2))))