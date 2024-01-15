import numpy as np
import matplotlib.pyplot as plt
import math
import copy

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
        return 1 / (r*(1/500)+1)

    @staticmethod
    def c2(r):
        ''' Mathematical function exp(-x) '''
        return np.exp(-r)

    @staticmethod
    def c3(r):
        ''' 1 - phi, with phi the c.d.f of a gaussian random variable following a N(5, 0.2) (which have the properties c(r) --> 1 when r --> 0 (instead of c(0)=1) and c(r) --> 0 when r --> +inf)'''
        if r == 0:
            return 1
        # here u corresponds to the distance from the center of data being kept
        u = 100
        # s corresponds here to the width of the "blur" data, i.e : the width of the blur bound at the edge of the circle of low pressure"
        # link with the density function of an N(u, s²)
        s = np.sqrt(10)
        return 1 - (1/2 * (1 + math.erf((r - u)/(s*np.sqrt(2)))))
    
    @staticmethod
    def c4(r):
        ''' Sigmoid function (logistic function) : 1 / (1 + exp(-x)) '''
        if r == 0:
            return 1
        
        return 1 - (1 / (1 + np.exp(-(r+100))))
    
    @staticmethod
    def c5(r):
        return 1/(1+np.exp((r-1)*50))
    
    @staticmethod
    def simulate_low_pressure(img, xc, yc, function):
        """
            Takes an image as parameter and simulate low pressure applying 
            a decreasing function around the center of the fingerprint.
            Returns a copy of the image transformed.
        """
        # Create a copy of the image
        new_img = copy.deepcopy(img)

        # Ratio of the image
        ratio = ((img.n + img.m)/2)*0.3

        # For each pixels of the image
        for x in range(img.n):
            for y in range(img.m):

                # Compute the distance between (x,y) and (xc,yc)
                dist = Main_Course_1.distance_between_pixels(np.array([x,y*1.5]), np.array([xc,yc*1.5]))

                # Compute the new intensity of (x,y) pixel according to its distance from the center
                new_intensity = function(dist/ratio)
                new_img.data[x][y] = 1-new_intensity*(1-img.data[x][y])

        return new_img
    
if __name__ == "__main__":

    from image import Image
     
    img = Image("images/clean_finger.png")
    img.display()

    low_pressure_img = Main_Course_1.simulate_low_pressure(img, 200, 125, Main_Course_1.c5)
    low_pressure_img.display()

    low_pressure_img.binarize(low_pressure_img.compute_threshold())
    low_pressure_img.erosion()
    low_pressure_img.dilation("Horizontal Rectangle", 3)
    low_pressure_img.display()

    # low_pressure_img.blur(5)
    # low_pressure_img.display()