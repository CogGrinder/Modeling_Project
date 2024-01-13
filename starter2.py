import numpy as np
import matplotlib.pyplot as plt
import cv2


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
                - image : bi-dimensionnal array of size n x m retpresenting the pixel intensity of the image 
            Return tha value of the point '''
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
        v13 = Starter_2.linear_interp(point[0], x1_0, x3_0, image.intensity_of_center(x1), image.intensity_of_center(x3), image.n)
        v24 = Starter_2.linear_interp(point[0], x2_0, x4_0, image.intensity_of_center(x2), image.intensity_of_center(x4), image.n)
        
        # secondly, we compute the linear interpolation according to the horizontal axis, with the value obtained above
        return Starter_2.linear_interp(point[1], x1_1, x2_1, v13, v24, image.m)
    
    @staticmethod
    def pixel_center(i, j):
        ''' Return the exact value of the center of the pixel of coordinates (i,j) '''
        return np.array([int(i+0.5), int(j+0.5)])
    
    @staticmethod
    def unit_test_bilinear_interp():
        ''' Unit test to test bilinear_interp method '''
        
        # Create a 2x2 NumPy array with pixel values 0, 85, 170 and 255
        image_data = np.array([[0, 85], [170, 255]], dtype=np.uint8)
        # Save the NumPy array as an image file
        cv2.imwrite('image_unit_test_bilinear_interp.png', image_data)
        # open the test image created
        test_image = Image("image_unit_test_bilinear_interp.png")
        
        # let's have four test points
        y1 = (0.27, 1.34) # one point outside the square of pixel centers on the vertical axis
        y2 = (0.81, 1.05) # one point in the square of pixel centers
        y3 = (0.81, 1.78) # one point outside the square of pixel centers on the horizontal axis
        y4 = (1.91, 1.66) # one point outside the square of pixel centers, on both axis
        
        # true results computed by hand
        true_result_y1 = 0.28
        true_result_y2 = 0.39
        true_result_y3 = 0.54 
        true_result_y4 = 1
        test_result_y1 = Starter_2.bilinear_interp(y1, test_image)
        test_result_y2 = Starter_2.bilinear_interp(y2, test_image)
        test_result_y3 = Starter_2.bilinear_interp(y3, test_image)
        test_result_y4 = Starter_2.bilinear_interp(y4, test_image)
                
        if true_result_y1 == test_result_y1 and true_result_y2 == test_result_y2 and true_result_y3 == test_result_y3 and true_result_y4 == test_result_y4:
            print("Unit test for bilinear interpolation method succesfull")
            return True
        else:
            print("Error in unit test for bilinear interpolation method")
            return False
        


if __name__ == "__main__":

    from image import Image
    
    Starter_2.unit_test_bilinear_interp()

    # img = Image("images/clean_finger.png")
    # img.display()

    # img.rotate_translate(16, (100, 100), (-50,15))

    # img.display()

    # img.rotate_translate(-16, (100, 100), (50,-15))

    # img.display()
