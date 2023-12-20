import numpy as np
import cv2
import matplotlib.pyplot as plt


class Image:
    def __init__(self, filename):
        self.__data = cv2.imread(filename, 0)
        self.__normalize()

    def display(self):
        """
            Display the image
        """
        plt.imshow(self.__data, cmap='gray')
        plt.show()

    def save(self, filename):
        """
            Save the image into a file
        """
        self.__denormalize()
        cv2.imwrite(filename, self.__data)
        self.__normalize()

    def max(self):
        """
            Returns the maximum intensity of the image
        """
        return np.max(self.__data)

    def min(self):
        """
            Returns the minimum intensity of the image
        """
        return np.min(self.__data)

    def __normalize(self):
        """
            Create and return a normalized version of the image
            The minimum value of image becomes 0 and its maximum value becomes 1
        """
        min = self.min()
        max = self.max()
        self.__data = (self.__data - min)/(max - min)

    def __denormalize(self, original_min=-1, original_max=-1):
        """ 
            Create and return the non normalized version (with pixels value living in [0;255], not in [0;1]) of the normalized image img
            Parameters original_min and original_max allows to not lose any information in comparison with the original version
            Those parameters are initially set to -1, which means that the denormalized image will pixel values in the full interval [0;255]
            (0 will be transformed into 0, 1 into 255) 
        """
        if original_max == -1 and original_min == -1:
            self.__data *= 256
        else:
            self.__data = self.__data * \
                (original_max - original_min) + original_min
        self.__data = self.__data.astype(int)

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

        self.__data[corner[0]: corner[0] + width, corner[1]: corner[1] + length] = value

    def symmetry(self, axis = 0):
        ''' Create and return the symetric of img with respect to the y axis '''
        n, m = self.__data.shape
        tmp=np.copy(self.__data)
        for x in range(n):
            for y in range(m):
                if axis == 0:
                    self.__data[x][y] = tmp[n - 1 - x][y]
                else:
                    self.__data[x][y] = tmp[x][m - 1 - y]

    def rotate(self, p, center_normalized):
        ''' 2D rotation of the img matrix in a p angle
        Makes the image bigger to compensate'''
        center = np.array([self.__data.shape[0]//(1/center_normalized[0]),self.__data.shape[1]//(1/center_normalized[1])]).astype(int)
        print(center)
        p_radian = p * np.pi/180
        n, m = self.__data.shape
        rotation_matrix = np.array([[np.cos(p_radian), -np.sin(p_radian)],[np.sin(p_radian), np.cos(p_radian)]])
        
        n_rotated = n
        m_rotated = m
        offset_x = 0
        offset_y = 0
        for i_c in [0,n-1] :
            for j_c in [0,m-1] :
                i = i_c-center[0]
                j = j_c-center[1]
                index_centered = np.array([i, j])
                rotated_ind_centered = np.floor(rotation_matrix @ index_centered).astype(int)
                rotated_ind = rotated_ind_centered + center
                
                if rotated_ind[0]>n_rotated :
                    n_rotated = rotated_ind[0]
                elif rotated_ind[0] < -offset_x:
                    offset_x = -rotated_ind[0]
                
                if rotated_ind[1]>m_rotated :
                    m_rotated = rotated_ind[1]
                elif rotated_ind[1] < -offset_y:
                    offset_y = -rotated_ind[1]

        print(offset_x,offset_y)
        print(n_rotated,m_rotated)

        tmp = np.copy(self.__data)
        self.__data = np.ones((n_rotated + offset_x, m_rotated + offset_y))

        for i_c in range(0,n_rotated + offset_x):
            for j_c in range(0,m_rotated + offset_y):
                i = i_c-center[0] - offset_x
                j = j_c-center[1] - offset_y
                index_centered = np.array([i, j])
                rotated_ind = np.floor(rotation_matrix @ index_centered).astype(int)
                # print(rotated_ind)
                if (0 <= rotated_ind[0] + center[0] < n) and (0 <=rotated_ind[1] + center[1]  < m):
                    self.__data[i_c][j_c] = tmp[rotated_ind[0] + center[0] ][rotated_ind[1] + center[1]]


    def rotate2(self, p, center_normalized):
        ''' 2D rotation of the img matrix in a p angle
         Keeps the image size constant '''
        center = np.array([self.__data.shape[0]//(1/center_normalized[0]),self.__data.shape[1]//(1/center_normalized[1])]).astype(int)
        print(center)
        p_radian = p * np.pi/180
        n, m = self.__data.shape
        rotation_matrix = np.array([[np.cos(p_radian), -np.sin(p_radian)],[np.sin(p_radian), np.cos(p_radian)]])
        

        tmp = np.copy(self.__data)
        self.__data = np.ones((n,m))

        for i_c in range(0,n):
            for j_c in range(0,m):
                i = i_c-center[0]
                j = j_c-center[1]
                index_centered = np.array([i, j])
                rotated_ind = np.floor(rotation_matrix @ index_centered).astype(int)
                # print(rotated_ind)
                if (0 <= rotated_ind[0] + center[0] < n) and (0 <=rotated_ind[1] + center[1]  < m):
                    self.__data[i_c][j_c] = tmp[rotated_ind[0] + center[0] ][rotated_ind[1] + center[1]]

    def blur(self, kernel_size):
        """
            Convolve the image with a blur kernel which size is passed by argument
            TODO: Implement 2D convolution from scratch
        """
        # Definition of a blur kernel
        k = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

        # Convolve the image with a blur kernel
        self.__data = cv2.filter2D(src=self.__data, ddepth=-1, kernel=k)

    def conv_2d(self, kernel):
        N = kernel.shape[0]
        n,m = self.__data.shape

        # Padded version with edge of the image
        padded_image = np.pad(self.__data, (N-1)//2, mode='constant')


        self.__data = np.zeros((n,m))
        
        for i in range(0, n):
            for j in range(0, m):
                result = 0
                for N_i in range(-(N-1)//2, (N-1)//2 + 1):
                    for N_j in range(-(N-1)//2, (N-1)//2 + 1):
                        result += kernel[N_i + (N-1)//2 ][N_j + (N-1)//2] * padded_image[i-N_i][j-N_j]
                self.__data[i][j] = result