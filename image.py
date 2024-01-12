import numpy as np
import math
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import copy

from starter2 import Starter_2
from starter3 import Starter_3
# from main_course_1 import Main_Course_1

class Image:
    def __init__(self, filename):
        self.data = cv2.imread(filename, 0)
        self.n, self.m = self.data.shape
        self.normalize()

    def display(self):
        """
            Display the image
        """
        plt.imshow(self.data, cmap='gray', vmin=0, vmax=1)
        plt.show()

    def save(self, filename):
        """
            Save the image into a file
        """
        self.__denormalize()
        cv2.imwrite(filename, self.data)
        self.normalize()

    def max(self):
        """
            Returns the maximum intensity of the image
        """
        return np.max(self.data)

    def min(self):
        """
            Returns the minimum intensity of the image
        """
        return np.min(self.data)

    def normalize(self):
        """
            Create and return a normalized version of the image
            The minimum value of image becomes 0 and its maximum value becomes 1
        """
        min = self.min()
        max = self.max()
        self.data = (self.data - min)/(max - min)

    def __denormalize(self, original_min=-1, original_max=-1):
        """ 
            Create and return the non normalized version (with pixels value living in [0;255], not in [0;1]) of the normalized image img
            Parameters original_min and original_max allows to not lose any information in comparison with the original version
            Those parameters are initially set to -1, which means that the denormalized image will pixel values in the full interval [0;255]
            (0 will be transformed into 0, 1 into 255) 
        """
        if original_max == -1 and original_min == -1:
            self.data *= 256
        else:
            self._data = self._data * \
                (original_max - original_min) + original_min
        self.data = self.data.astype(int)

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

    def symmetry(self, axis=0):
        ''' Return the symetric of img with respect to the x axis if axis=0,
                                                    to the y axis if axis=1 '''
        tmp=np.copy(self.data)
        for x in range(self.n):
            for y in range(self.m):
                if axis == 0:
                    self.data[x][y] = tmp[self.n - 1 - x][y]
                else:
                    self.data[x][y] = tmp[x][self.m - 1 - y]

    def symmetry_diagonal(self, axis=0):
        ''' Return the symmetric of the image with respect to the diagonal going from bottom left corner to top right corner if axis=0
                                                                           going from top left corner to bottom right corner if axis=1 '''
        tmp = np.copy(self.data)
        self.data = np.ones((self.m, self.n))
        for x in range(self.n):
            for y in range(self.m):
                if axis == 0:
                    self.data[x][y] = tmp[y][x]
                else:
                    self.data[x][y] = tmp[self.n - y][self.m - x]


    def simulate_low_pressure(self, center, c):
        ''' Return the image at which we have simulate a low pressure of center center.
            Parameters :
                - center : coordinates of the pixel center of the low pressure (tuple of two int values)
                - c : mathematical function of one argument (c(r)), monotonically decreasing as r tends to infinity, with c(0)=1 and c(r)=0 the limit when
                r tends to infinity.
        '''
        center_coord = Starter_2.pixel_center(center[0], center[1])
        for x in range(self.n):
            for y in range(self.m):
                distance = Main_Course_1.distance_between_pixels((x, y), center_coord)
                self.data[x][y] *= c(distance)


    def rotate_translate(self, p, center, offset):
        ''' Complete the rotation-translation operation on the image
            Parameters :
                - p : rotation angle, in degree
                - center : rotation center, tuple of two int values (supposed to be contained in the image shape), eg: (150, 200), for an image of shape 300x500
                - offset : parameters of the translation, tuple of int values, eg: (2, -3) --> translation : (x', y') = (x + 2, y - 3)
        '''
        # create a deepcopy of the self instance
        tmp = copy.deepcopy(self)
        self.data = np.ones((self.n,self.m))
        
        # Part 1 : perform the rotation
        # Convert p to radian
        p_radian = p * np.pi/180
        # compute the inverse rotatation matrix
        inverse_rotation_matrix = np.array([[np.cos(p_radian), np.sin(p_radian)],[-np.sin(p_radian), np.cos(p_radian)]])
        # center of rotation coordiantes (coordinates of the center of the pixel "center", given as parameter)
        coord_center_of_rotation = Starter_2.pixel_center(center[0], center[1])
        for i in range(0, self.n):
            for j in range(0, self.m):
                # for each pixel of the result image, calculate its coordinates by the inverse rotation matrix
                # get the coordinates of the center of the pixel
                pixel_center = Starter_2.pixel_center(i, j)
                # adapt coordinates to the center of rotation
                pixel_to_rotate = np.array([pixel_center[0] - coord_center_of_rotation[0], pixel_center[1] - coord_center_of_rotation[1]])
                # calculate the inverse image by the rotation
                inverse_coord = np.dot(inverse_rotation_matrix, pixel_to_rotate)
                if (0 <= inverse_coord[0] + coord_center_of_rotation[0] <= self.n) and (0 <= inverse_coord[1] + coord_center_of_rotation[1] <= self.m):
                    # if the coordinates of the pixel by the inverse rotation matrix is in the range of the original image
                    # let's perform a bi-linear interpolation to compute the intensity of the rotated pixel
                    self.data[i][j] = Starter_2.bilinear_interp(np.array([inverse_coord[0] + coord_center_of_rotation[0], inverse_coord[1] + coord_center_of_rotation[1]]), tmp)
                # otherwise the pixel intensity is set to 1 
        # free up memory space occupied by tmp
        del tmp

        # Part 2 : perform the translation
        # create a deepcopy of the self instance
        tmp = copy.deepcopy(self)
        self.data = np.ones((self.n,self.m))
        for i in range(0, self.n):
            for j in range(0, self.m):
                if (0 <= i - offset[0] < self.n) and (0 <= j - offset[1] < self.m):
                    self.data[i][j] = tmp.data[i - offset[0]][j - offset[1]]
                else:
                    self.data[i][j] = 1
        # free up memory space occupied by tmp
        del tmp

    def intensity_of_center(self, point):
        ''' Return the pixel intensity of the pixel of center point=(i, j) '''
        if (0 <= point[0]-0.5 < self.n) and (0 <= point[1]-0.5 < self.m):
            return self.data[int(point[0]-0.5)][int(point[1]-0.5)]
        else:
            return 1
       
    def blur(self, kernel_size):
        """
            Convolve the image with a blur kernel which size is passed by argument
            TODO: Implement 2D convolution from scratch
        """
        # Definition of a blur kernel
        k = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

        # Convolve the image with a blur kernel
        self.data = cv2.filter2D(src=self.data, ddepth=-1, kernel=k)

    def conv_2d(self, xc, yc):
        
        n,m = self.data.shape

        # Padded version with edge of the image
        # we chose 7 because 7 = 15 // 2
        # padded_image = np.pad(self._data, (N-1)//2, mode='constant')
        padded_image = np.pad(self.data, 7, mode='edge')

        self.data = np.zeros((n,m))
        
        for i in range(0, n):
            for j in range(0, m):
                K = Starter_3.kernel(i,j,xc,yc,self.m,self.n)
                N = K.shape[0]
                result = 0
                for N_i in range(-(N-1)//2, (N-1)//2 + 1):
                    for N_j in range(-(N-1)//2, (N-1)//2 + 1):
                        # we chose 7 because 7 = 15 // 2
                        result += K[N_i + (N-1)//2 ][N_j + (N-1)//2] * padded_image[i-N_i + 7][j-N_j + 7]
                self.data[i][j] = result

    def fft_2d(self):
        """
            Return an array of the 2D fast Fourier transform applied on the image
        """
        ft = np.fft.ifftshift(self.data)
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
        self.data = signal.fftconvolve(self.data, g, mode="same")

    # def test_black(self, n=5):
    #     np.set_printoptions(precision=1)
    #     self._data = np.ones((n,n))