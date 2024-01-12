import numpy as np
import math
from scipy import signal
import cv2
import matplotlib.pyplot as plt

from starter3 import Starter_3

class Image:
    def __init__(self, filename):
        self._data = cv2.imread(filename, 0)
        self.__n, self.__m = self._data.shape
        self.__normalize()

    def display(self):
        """
            Display the image
        """
        plt.imshow(self._data, cmap='gray')
        plt.show()

    def save(self, filename):
        """
            Save the image into a file
        """
        self.__denormalize()
        cv2.imwrite(filename, self._data)
        self.__normalize()

    def max(self):
        """
            Returns the maximum intensity of the image
        """
        return np.max(self._data)

    def min(self):
        """
            Returns the minimum intensity of the image
        """
        return np.min(self._data)

    def __normalize(self):
        """
            Create and return a normalized version of the image
            The minimum value of image becomes 0 and its maximum value becomes 1
        """
        min = self.min()
        max = self.max()
        self._data = (self._data - min)/(max - min)

    def __denormalize(self, original_min=-1, original_max=-1):
        """ 
            Create and return the non normalized version (with pixels value living in [0;255], not in [0;1]) of the normalized image img
            Parameters original_min and original_max allows to not lose any information in comparison with the original version
            Those parameters are initially set to -1, which means that the denormalized image will pixel values in the full interval [0;255]
            (0 will be transformed into 0, 1 into 255) 
        """
        if original_max == -1 and original_min == -1:
            self._data *= 256
        else:
            self._data = self._data * \
                (original_max - original_min) + original_min
        self._data = self._data.astype(int)

    def create_rectangle(self, corner, width, length, color):
        ''' Create and return a version of the image, where has been included a rectangle of origin (top-left corner) origin (tuple of 2 int values, coordinates
        of the origin), of width witdth and lenght lenght, with pixel value value (normalized) '''
        match color:
            case "black":
                value = 0
            case "white":
                value = 1
            case _:
                raise ValueError("invalid color")

        self._data[corner[0]: corner[0] + width, corner[1]: corner[1] + length] = value

    def symmetry(self, axis = 0):
        ''' Create and return the symetric of img with respect to the y axis '''
        n, m = self._data.shape
        tmp=np.copy(self._data)
        for x in range(n):
            for y in range(m):
                if axis == 0:
                    self._data[x][y] = tmp[n - 1 - x][y]
                else:
                    self._data[x][y] = tmp[x][m - 1 - y]

    def linear_interp(self, x, x1, x2, vx1, vx2):
        ''' Perorm the linear interpolation between the points x1 and x2, of values vx1 and vx2 
        NB : we have ||x1 - x2|| = 1
            Parameters:
                - x : point of which we want to calculate the value threw interpolation (we assume that x is in [x1, x2])
                - x1, x2 : points from which we know the values
                - vx1, vx2 : values at points x1 and x2
            Return the value vx, of the point x  '''
        # Cases where x belongs to the surroundings (of margin length 0.5) of the image
        if x1 < 0:
            return vx2
        if x2 > self.__n:
            return vx1
        
        alpha = x - x1
        beta = 1 - alpha
        return vx1 * (1 - alpha) + vx2 * (1 - beta)

    def bilinear_interp(self, point, image):
        ''' Perform the bilinear interpolation of the coordinate point in the image image 
            Parameters :
                - point : tuple of float, belonging to [0;n]x[0;m]
                - image : bi-dimensionnal of size n x m retpresenting the pixel intensity of the image '''
        # find the four points to perform the bi-linear interpolation
        if point[0] - np.floor(point[0]) > 0.5:
            x1_0 = np.floor(point[0]) + 0.5
            x3_0 = np.ceil(point[0]) + 0.5
        else:
            x1_0 = np.floor(point[0]) - 0.5
            x3_0 = np.floor(point[0]) + 0.5
        x2_0 = x1_0
        x4_0 = x3_0
        
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
        v13 = self.linear_interp(point[0], x1_0, x3_0, self.intensity_of_center(x1), self.intensity_of_center(x3))
        v24 = self.linear_interp(point[0], x2_0, x4_0, self.intensity_of_center(x2), self.intensity_of_center(x4))
        
        # secondly, we compute the linear interpolation according to the horizontal axis, with the value obtained above
        return self.linear_interp(point[1], x1_1, x2_1, v13, v24)


    def rotate_translate(self, p, center, offset):
        ''' Complete the rotation-translation operation on the image
            Parameters :
                - p : rotation angle, in degree
                - center : rotation center, tuple of two int values (supposed to be contained in the image shape), eg: (150, 200), for an image of shape 300x500
                - offset : parameters of the translation, tuple of int values, eg: (2, -3) --> translation : (x', y') = (x + 2, y - 3)
        '''

        # temporary copy of the grid
        tmp = np.copy(self._data)
        self._data = np.ones((self.__n,self.__m))

        # Part 1 : perform the rotation
        # Convert p to radian
        p_radian = p * np.pi/180
        # compute the inverse rotatation matrix
        inverse_rotation_matrix = np.array([[np.cos(p_radian), np.sin(p_radian)],[-np.sin(p_radian), np.cos(p_radian)]])
        for i in range(0, self.__n):
            for j in range(0, self.__m):
                # for each pixel of the result image, calculate its coordinates by the inverse rotation matrix
                pixel_center = self.pixel_center(i, j)
                inverse_coord = np.dot(inverse_rotation_matrix, pixel_center)
                # perform a bi-linear interpolation to compute the intensity of the rotated pixel
                new_intensity = self.bilinear_interp(inverse_coord, tmp)

    @staticmethod
    def pixel_center(i, j):
        ''' Return the exact value of the center of the pixel of coordinates (i,j) '''
        return np.array([int(i+0.5), int(j+0.5)])
    
    @staticmethod
    def intensity_of_center(point):
        ''' Return the pixel intensity of the pixel of center point=(i, j) '''
        return self._data[int(point[0]-0.5), int(point[1]-0.5)]

    def blur(self, kernel_size):
        """
            Convolve the image with a blur kernel which size is passed by argument
            TODO: Implement 2D convolution from scratch
        """
        # Definition of a blur kernel
        k = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

        # Convolve the image with a blur kernel
        self._data = cv2.filter2D(src=self._data, ddepth=-1, kernel=k)

    def conv_2d(self, xc, yc):
        
        n,m = self._data.shape

        # Padded version with edge of the image
        # we chose 7 because 7 = 15 // 2
        # padded_image = np.pad(self._data, (N-1)//2, mode='constant')
        padded_image = np.pad(self._data, 7, mode='edge')

        self._data = np.zeros((n,m))
        
        for i in range(0, n):
            for j in range(0, m):
                K = Starter_3.kernel(i,j,xc,yc,self.__m,self.__n)
                N = K.shape[0]
                result = 0
                for N_i in range(-(N-1)//2, (N-1)//2 + 1):
                    for N_j in range(-(N-1)//2, (N-1)//2 + 1):
                        # we chose 7 because 7 = 15 // 2
                        result += K[N_i + (N-1)//2 ][N_j + (N-1)//2] * padded_image[i-N_i + 7][j-N_j + 7]
                self._data[i][j] = result

    def fft_2d(self):
        """
            Return an array of the 2D fast Fourier transform applied on the image
        """
        ft = np.fft.ifftshift(self._data)
        ft = np.fft.fft2(ft)
        return np.fft.fftshift(ft)
    
    def ifft_2d(self, ft):
        """
            Return an array of the 2D inverse fast Fourier transform
        """
        ift = np.fft.ifft2(ft)
        return np.fft.fftshift(ift)
    
    def fft_conv_2d(self, g):
        """
            Convolve the image using the relation between convolution product and Fourier transform 
            i.e. f*g = IFFT( FFT(f).FFT(g) )
        """
        # ft_f = self.fft_2d()
        # ft_g = np.fft.fftshift(np.fft.fft2(g))
        # print("ft_f:", ft_f.shape)
        # print("ft_g:", ft_g.shape)
        # ift_fg = np.fft.ifft2(np.multiply(ft_f, ft_g))
        # return np.fft.fftshift(ift_fg)
        self._data = signal.fftconvolve(self._data, g, mode="same")

    # def test_black(self, n=5):
    #     np.set_printoptions(precision=1)
    #     self._data = np.ones((n,n))