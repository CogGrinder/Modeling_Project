import numpy as np
import cv2
import matplotlib.pyplot as plt

from image import Image

class Starter_2:

    @staticmethod
    def linear_interp(x, x1, x2, vx1, vx2, bound_sup):
        ''' Perorm the linear interpolation between the points x1 and x2, of values vx1 and vx2 
        NB : we have ||x1 - x2|| = 1
            Parameters:
                - x : point of which we want to calculate the value threw interpolation (we assume that x is in [x1, x2])
                - x1, x2 : points from which we know the values
                - vx1, vx2 : values at points x1 and x2
                - bound_sup : int, taking two possible values : either n or m (maximum size of x2), in order to know if the highest bound of x2 is n or m
            Return the value vx, of the point x  '''
        # Cases where x belongs to the surroundings (of margin length 0.5) of the image
        if x1 < 0:
            return vx2
        if x2 > bound_sup:
            return vx1
        
        alpha = x - x1
        beta = 1 - alpha
        return vx1 * beta + vx2 * alpha # since we are in the case where ||x1 - x2|| = 1, then this expression is equivalent to : vx1 * (1-alpha) + vx2 * (1-beta)

    @staticmethod
    def bilinear_interp(point, image):
        ''' Perform the bilinear interpolation of the coordinate point in the image image 
            Parameters :
                - point : tuple of float, belonging to [0;n]x[0;m]
                - image : bi-dimensionnal array of size n x m retpresenting the pixel intensity of the image '''
        # find the four points to perform the bi-linear interpolation
        # find their vertical axis value
        if point[0] - np.floor(point[0]) > 0.5:
            x1_0 = np.floor(point[0]) + 0.5
            x3_0 = np.ceil(point[0]) + 0.5
        else:
            x1_0 = np.floor(point[0]) - 0.5
            x3_0 = np.floor(point[0]) + 0.5
        x2_0 = x1_0
        x4_0 = x3_0
        
        # find their horizontal axis value
        if point[1] - np.floor(point[1]) > 0.5:
            x1_1 = np.floor(point[1]) + 0.5
            x2_1 = np.ceil(point[1]) + 0.5
        else:
            x1_1 = np.floor(point[1]) - 0.5
            x2_1 = np.floor(point[1]) + 0.5
        x3_1 = x1_1
        x4_1 = x2_1

        x1 = np.array([x1_0, x1_1])
        x2 = np.array([x2_0, x2_1])
        x3 = np.array([x3_0, x3_1])
        x4 = np.array([x4_0, x4_1])

        # first, we compute the linear interpolation according to the vertical axis
        v13 = Starter_2.linear_interp(point[0], x1_0, x3_0, image.intensity_of_center(x1), image.intensity_of_center(x3), image.__n)
        v24 = Starter_2.linear_interp(point[0], x2_0, x4_0, image.intensity_of_center(x2), image.intensity_of_center(x4), image.__n)
        
        # secondly, we compute the linear interpolation according to the horizontal axis, with the value obtained above
        return Starter_2.linear_interp(point[1], x1_1, x2_1, v13, v24, image.__m)
    
    @staticmethod
    def pixel_center(i, j):
        ''' Return the exact value of the center of the pixel of coordinates (i,j) '''
        return np.array([int(i+0.5), int(j+0.5)])






if __name__ == "__main__":

    img = Image("images/clean_finger.png")
    img.display()

    img.rotate_translate(16, (100, 100), (-50,15))

    img.display()

    img.rotate_translate(-16, (100, 100), (50,-15))

    img.display()
