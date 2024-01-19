import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
from pathlib import Path #for extracting name from filename

from starter2 import Starter_2
from starter3 import Starter_3
from main_course_1 import Main_Course_1

class Image:
    def __init__(self, filename):
        self.data = cv2.imread(filename, 0)
        self.n, self.m = self.data.shape
        self.name = Path(filename).stem
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
        
    def add_rows_of_pixels(self, num_rows=1, at_beginning=True):
        """
        Add rows of ones to the data array.

        Parameters:
        - num_rows: Number of lines to add.
        - at_beginning: If True, add rows at the beginning; otherwise, add at the end.
        """
        ones_rows = np.ones((num_rows, self.m))
        if at_beginning:
            self.data = np.vstack([ones_rows, self.data])
        else:
            self.data = np.vstack([self.data, ones_rows])
            
        # Update the number of rows (self.n) after adding lines
        self.n += num_rows
        
        
    def add_columns_of_pixels(self, num_columns=1, at_beginning=True):
        """
        Add columns of ones to the data array.

        Parameters:
        - num_columns: Number of columns to add.
        - at_beginning: If True, add columns at the beginning; otherwise, add at the end.
        """
        ones_columns = np.ones((self.n, num_columns), dtype=np.uint8)
        if at_beginning:
            self.data = np.hstack([ones_columns, self.data])
        else:
            self.data = np.hstack([self.data, ones_columns])

        # Update the number of columns (self.m) after adding columns
        self.m += num_columns

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
            self.data = self.data * \
                (original_max - original_min) + original_min
        self.data = self.data.astype(int)

    def create_rectangle(self, corner, width, length, color):
        ''' Create and return a version of the image, where has been included a rectangle of origin (top-left corner) origin (tuple of 2 int values, coordinates
        of the origin), of width witdth and lenght lenght, with pixel value value (normalized) '''
        if (color == "black"):
            value = 0
        elif (color == "white"):
            value = 1
        else:
            raise ValueError("invalid color")

        self.data[corner[0]: corner[0] + width, corner[1]: corner[1] + length] = value

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


    # def simulate_low_pressure(self, center, c):
    #     ''' Return the image at which we have simulate a low pressure of center center.
    #         Parameters :
    #             - center : coordinates of the pixel center of the low pressure (tuple of two int values)
    #             - c : mathematical function of one argument (c(r)), monotonically decreasing as r tends to infinity, with c(0)=1 and c(r)=0 the limit when
    #             r tends to infinity.
    #     '''
    #     center_coord = Starter_2.pixel_center(center[0], center[1])
    #     for x in range(self.n):
    #         for y in range(self.m):
    #             distance = Main_Course_1.distance_between_pixels((x, y), center_coord)
    #             self.data[x][y] = 1 - ((1-self.data[x][y])*c(distance)) # mulptiply (1 - pixel_value) by c(distance) and then substract the obteined value to 1 and not multiplying c(distance) directly since we want the image to become whiter (so close to 1) and not darker


    def rotate_translate(self, p, center, offset, data_conservation=False):
        ''' Complete the rotation-translation operation on the image (perform the rotation and then, the translation !!! Operation not commutative !)
            Parameters :
                - p : rotation angle, in degree
                - center : rotation center, tuple of two int values (supposed to be contained in the image shape), eg: (150, 200), for an image of shape 300x500
                - offset : parameters of the translation, tuple of int values, eg: (2, -3) --> translation : (x', y') = (x + 2, y - 3)
                - data_conservation : boolean, if true, then no data is lost during the transformation, meaning that the imga is redimensionned to keep all the data of the original image in the transformed image. Set to False by default
        '''     
        # create a deepcopy of the self instance
        tmp = copy.deepcopy(self)
        self.data = np.ones((self.n,self.m)) 
           
        # Convert p to radian
        p_radian = p * np.pi/180
        # center of rotation coordinates (coordinates of the center of the pixel "center", given as parameter)
        coord_center_of_rotation = Starter_2.pixel_center(center[0], center[1])
        
        # range of pixels to browse in the transformed image
        # if no data conservation, it's image size
        min_vert_range = 0
        min_hori_range = 0
        max_vert_range = self.n
        max_hori_range = self.m
        origin = np.array([0, 0])
        
        # image size redimensionning in case of data conservation
        if data_conservation:
            # calculate the rectangle (oriented according to the axis --> here it's a possibility to improve : get a data frame that fits more the data than a rectangle axis oriented) that contains all the data
            # i.e : calculate min and max index, for both axis, in which all the information is contained
            # browse all the image
            min_vert = self.n
            min_hori = self.m
            max_vert = 0
            max_hori = 0
            for i in range(0, self.n):
                for j in range(0, self.m):
                    # if there is information, then pixel value is different than white (i.e : value below 1)
                    if tmp.data[i][j] < 1:
                        if i < min_vert:
                            min_vert = i
                        if i > max_vert:
                            max_vert = i
                        if j < min_hori:
                            min_hori = j
                        if j > max_hori:
                            max_hori = j
            # calculate the coordinates of the four vertices
            x1 = Starter_2.pixel_center(min_vert, min_hori)
            x2 = Starter_2.pixel_center(min_vert, max_hori)
            x3 = Starter_2.pixel_center(max_vert, min_hori)
            x4 = Starter_2.pixel_center(max_vert, max_hori)
            # calculate the image of the rectangle by the transformation 
            # compute the matrix of the transformation
            # using 3x3 matrix to be able to make the product of matrix to compose transformations (homogeneous coordinates)
            translation_matrix = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])
            rotation_matrix = np.array([[np.cos(p_radian), -np.sin(p_radian), (1-np.cos(p_radian)) * coord_center_of_rotation[0] + coord_center_of_rotation[1] * np.sin(p_radian)], [np.sin(p_radian), np.cos(p_radian), -coord_center_of_rotation[0] * np.sin(p_radian) + coord_center_of_rotation[1] * (1-np.cos(p_radian))], [0, 0, 1]])
            transformation_matrix = np.dot(translation_matrix, rotation_matrix)
            image_x1 = np.dot(transformation_matrix, np.array([x1[0], x1[1], 1]))
            image_x2 = np.dot(transformation_matrix, np.array([x2[0], x2[1], 1]))
            image_x3 = np.dot(transformation_matrix, np.array([x3[0], x3[1], 1]))
            image_x4 = np.dot(transformation_matrix, np.array([x4[0], x4[1], 1]))
            # find the extreme values of the image
            min_vert = int(np.floor(np.min(np.array([image_x1[0], image_x2[0], image_x3[0], image_x4[0]]))))
            max_vert = int(np.ceil(np.max(np.array([image_x1[0], image_x2[0], image_x3[0], image_x4[0]]))))
            min_hori = int(np.floor(np.min(np.array([image_x1[1], image_x2[1], image_x3[1], image_x4[1]]))))
            max_hori = int(np.ceil(np.max(np.array([image_x1[1], image_x2[1], image_x3[1], image_x4[1]]))))
            # redimension by adding blank data in the image if necessary
            if max_vert > self.n:
                self.add_rows_of_pixels(max_vert - self.n, at_beginning=False)
            if min_vert < 0:
                    self.add_rows_of_pixels(np.abs(min_vert), at_beginning=True)
                    # if we add rows at the beginning, the original origin is evolving like this (taking +np.abs(min_vert) rows):
                    origin[0] += np.abs(min_vert)
            if max_hori > self.m:
                self.add_columns_of_pixels(max_hori - self.m, at_beginning=False)
            if min_hori < 0:
                self.add_columns_of_pixels(np.abs(min_vert), at_beginning=True)  
                # if we add columns at the beginning, the original origin is evolving like this (taking +np.abs(min_hori) columns):
                origin[1] += np.abs(min_hori)
            
            # update of the range of pixels to browse in the transformed image
            min_vert_range = min_vert
            min_hori_range = min_hori
            max_vert_range =  max_vert
            max_hori_range = max_hori
            
        
        # calculate the inverse operation : (rotation --> translation)^(-1) = translation^(-1) --> rotation^(-1)
        # using 3x3 matrix to be able to make the product of matrix to compose transformations (homogeneous coordinates)
        inverse_translation_matrix = np.array([[1, 0, -offset[0]], [0, 1, -offset[1]], [0, 0, 1]])
        inverse_rotation_matrix = np.array([[np.cos(p_radian), np.sin(p_radian), (1-np.cos(p_radian)) * coord_center_of_rotation[0] - coord_center_of_rotation[1] * np.sin(p_radian)], [-np.sin(p_radian), np.cos(p_radian), coord_center_of_rotation[0] * np.sin(p_radian) + coord_center_of_rotation[1] * (1-np.cos(p_radian))], [0, 0, 1]])
        inverse_transformation_matrix = np.dot(inverse_rotation_matrix, inverse_translation_matrix)             
             
        # for each pixel of the finished image, calculate its counter image by the rotation translation
        for i in range(min_vert_range, max_vert_range):
            for j in range(min_hori_range, max_hori_range):
                # get the coordinates of the center of the pixel
                pixel_center = Starter_2.pixel_center(i, j)
                counter_image_coord = np.dot(inverse_transformation_matrix, np.array([pixel_center[0], pixel_center[1], 1]))
                if (0 <= counter_image_coord[0] <= tmp.n) and (0 <= counter_image_coord[1] <= tmp.m):
                    # if the coordinates of the pixel by the inverse rotation matrix is in the range of the original image
                    # let's perform a bi-linear interpolation to compute the intensity of the rotated pixel
                    self.data[i+origin[0]][j+origin[1]] = Starter_2.bilinear_interp(np.array([counter_image_coord[0], counter_image_coord[1]]), tmp)
                # otherwise the pixel intensity is set to 1 
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

        # Convert to float values to be sure that filter2() will process the data
        # it only handles float values
        self.data = self.data/1.0

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

    def test_black(self, n=5):
        np.set_printoptions(precision=1)
        self.data = np.ones((n,n))

    @staticmethod
    def otsu_intraclass_variance(image, threshold):
        """
        Otsu’s intra-class variance.
        If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
        """
        return np.nansum([np.mean(cls) * np.var(image, where=cls)
            #   weight   ·  intra-class variance
            for cls in [image>=threshold, image<threshold]
        ])
        # NaNs only arise if the class is empty, in which case the contribution should be zero, which `nansum` accomplishes.

   
    def compute_threshold(self):
        """
            Compute the threshold for binarization(See Method to select a threshold automatically from a gray level histogram, N. Otsu, 1975, Automatica.)
        """
        image = self
        image.__denormalize()
        
        otsu_threshold = min(
		    range( np.min(image.data)+1, np.max(image.data) ),
		    key = lambda th: self.otsu_intraclass_variance(image.data, th)
	    )  
        mini = self.min()
        maxi = self.max()
        otsu_threshold = (otsu_threshold- mini)/(maxi - mini)
        return otsu_threshold
    
    
    def image_hist(self):
        """
            Plot the grayscale histogram of the image
        """
        # create the histogram
        hist,bins = np.histogram(self.data, bins=256, range=(0, 1))
        # configure and draw the histogram figure
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixel count")
        plt.xlim([0.0, 1.0])  # <- named arguments do not work here

        plt.plot(bins[0:-1], hist)  # <- or here

  
    def binarize(self, threshold):
        """
            Binarize the image(pixels either 1 or 0) given a threshold
        """
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self.data[i][j] < threshold:
                    self.data[i][j] = 0
                else:
                    self.data[i][j] = 1


    def dilation(self, structuring_element = "Square", size = 3):
        """
            Dilate the binary version of an image
            
            params : 
                structuring element : Defined the shape of the structuring_element(geometrical shape) used to probe the image
                    Possible values : Square, Horizontal Rectangle, Vertical Horizontal

                size : Defined the size of the structuring element
        """
        if structuring_element=='Square':
            kernel = np.ones((size, size), np.uint8)
            orig_shape = self.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=self.data, pad_width=pad_width, mode='constant')
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (i == kernel).any() else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            self.data = image_dilate.reshape(orig_shape)
        
        if structuring_element=='Horizontal Rectangle':
            kernel = np.ones((2, size), np.uint8)
            orig_shape = self.data.shape
            pad_width = size - 1

            # pad the image with pad_width
            image_pad = np.pad(array=self.data, pad_width=pad_width, mode='constant')
            image_pad = image_pad[pad_width-1:image_pad.shape[0]-pad_width, :image_pad.shape[1]-pad_width]
            pimg_shape = image_pad.shape
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + 2), j:(j + size)]
                                         for i in range(pimg_shape[0] - 1) for j in range(pimg_shape[1] - size + 1)])
            
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (i == kernel).any() else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            self.data = image_dilate.reshape(orig_shape)
        
        if structuring_element=='Vertical Rectangle':
            kernel = np.ones((size, 2), np.uint8)
            orig_shape = self.data.shape
            pad_width = size - 1

            # pad the image with pad_width
            image_pad = np.pad(array=self.data, pad_width=pad_width, mode='constant')
            image_pad = image_pad[:image_pad.shape[0]-pad_width, pad_width-1:image_pad.shape[1]-pad_width]
            pimg_shape = image_pad.shape
            
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + 2)]
                                         for i in range(pimg_shape[0] - size + 1) for j in range(pimg_shape[1] - 1)])
            
            # replace the values either 1 or 0 by dilation condition
            image_dilate = np.array([1 if (i == kernel).any() else 0 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            self.data = image_dilate.reshape(orig_shape)
            
            for i in range(pad_width):
                self.data[:, i] = 1

         
    def erosion(self, structuring_element = "Square", size = 3):
        """
            Erode the binary version of an image
            
            params : 
                structuring element : Defined the shape of the structuring_element(geometrical shape) used to probe the image
                    Possible values : Square, Horizontal Rectangle, Vertical Horizontal

                size : Defined the size of the structuring element
        """
        if structuring_element=='Square':
            kernel = np.ones((size, size), np.uint8)
            orig_shape = self.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=self.data, pad_width=pad_width, mode='constant')
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            
            # replace the values either 1 or 0 by erosion condition
            image_erode = np.array([0 if (i != kernel).any() else 1 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            self.data = image_erode.reshape(orig_shape)

        if structuring_element=='Horizontal Rectangle':
            kernel = np.ones((2, size), np.uint8)
            orig_shape = self.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=self.data, pad_width=pad_width, mode='constant')
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + 2), j:(j + size)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            
            # replace the values either 1 or 0 by erosion condition
            image_erode = np.array([0 if (i != kernel).any() else 1 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            self.data = image_erode.reshape(orig_shape)

            for i in range(pad_width):
                self.data[:, i] = 1
                self.data[i+1, :] = 1
        
        if structuring_element=='Vertical Rectangle':
            kernel = np.ones((size, 2), np.uint8)
            orig_shape = self.data.shape
            pad_width = size - 2 

            # pad the image with pad_width
            image_pad = np.pad(array=self.data, pad_width=pad_width, mode='constant')
            pimg_shape = image_pad.shape
            h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
            
            # obtain the submatrices according to the size of the kernel
            flat_submatrices = np.array([image_pad[i:(i + size), j:(j + 2)]
                                         for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)])
            
            # replace the values either 1 or 0 by erosion condition
            image_erode = np.array([0 if (i != kernel).any() else 1 for i in flat_submatrices])
            # obtain new matrix whose shape is equal to the original image size
            self.data = image_erode.reshape(orig_shape)
            for i in range(pad_width):
                self.data[:, i] = 1
                self.data[i+1, :] = 1

    def crop_patches(self, n, size=9):
        """
            Collect randomly n patches of size x size pixels from the image and returns them
            within a list. 

            Parameters:
                - n: number of patches to crop
                - size : size of each patch (odd number, 9 by default)

            Returns:
                - list of patches (size x size numpy arrays)  
        """
        print("Cropping", n, "patches of size", size, "px from the image")

        # list of patches to be returned
        patches = []

        # crop n patches from the image
        for i in range(n):

            # generate randomly the top left corner index of a patch ranging in [0, n-size-1] x [0, m-size-1]
            top_left_coord_x = np.random.randint(0, self.n - size - 1)
            top_left_coord_y = np.random.randint(0, self.m - size - 1)

            # grab the corresponding patch of size x size from the image
            patch = self.data[top_left_coord_x : top_left_coord_x+size, top_left_coord_y : top_left_coord_y+size]
            patches.append(patch)

        # print(top_left_coord_x, top_left_coord_y)
        # print(patch)
        # plt.imshow(patch, cmap='gray', vmin=0, vmax=1)
        # plt.show()
            
        print("Cropping patches: Done.")
            
        return patches

            
             