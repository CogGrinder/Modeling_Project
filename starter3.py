import numpy as np
import math
from scipy import signal
import cv2
import matplotlib.pyplot as plt

class Starter_3:

    @staticmethod
    def kernel(i,j,xc,yc,m,n):
        """
            Returns a space dependent kernel used for motion blurring.  
        """

        # maximum distance from (i,j) point to each corners of the image
        dist_max = max(np.linalg.norm(np.array([i,j]) - np.array([0,0])),
                    np.linalg.norm(np.array([i,j]) - np.array([0,m])),
                    np.linalg.norm(np.array([i,j]) - np.array([n, 0])),
                    np.linalg.norm(np.array([i,j]) - np.array([n, m])))
        
        # we normalize distance to the center d((i,j), (x_c,y_c)) => (float between 0 and 1)
        normalized_dist = np.linalg.norm(np.array([i,j]) - np.array([xc,yc]))/dist_max

        # renormalize to the interval [1, 15] (15 is the max size of our kernel)
        N_float = normalized_dist * (15-1) + 1
        
        # round N_float to the previous odd integer
        N_odd = math.floor((N_float-1)/2)*2 + 1

        # create a N_odd + 2 size matrix
        matrix = np.eye(N_odd+2)[::-1]

        # fill diagonal corners
        matrix[0][-1] = (N_float - N_odd)/2 # normalization to [0,1]
        matrix[-1][0] = (N_float - N_odd)/2

        # preserve the intensity of the image
        return matrix/np.sum(matrix)

if __name__ == "__main__":

    from image import Image

    # kernel = np.ones((15, 15))/225
    img = Image("images/clean_finger.png")
    img.conv_2d(200, 125)
    img.display()